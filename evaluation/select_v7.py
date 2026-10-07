"""Lock V7 validation decisions and exact artifacts before opening final maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from env.v7_maps import TRAIN_LAYOUTS, VALIDATION_LAYOUTS, layout_hash


SEEDS = (41, 42, 43)
MAPS = tuple(VALIDATION_LAYOUTS)
EPISODES = 100


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(
    path: Path, expected_rows: int, identity_fields: tuple[str, ...], variants_per_game: int,
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, got {len(rows)}")
    identities = [tuple(row[field] for field in identity_fields) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError(f"Duplicate validation game in {path}")
    if {row["map"] for row in rows} != set(MAPS):
        raise ValueError(f"Validation map mismatch in {path}")
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["map"], row["training_seed"], row["eval_seed"])] += 1
    if len(counts) != len(MAPS) * EPISODES or any(
        count != variants_per_game for count in counts.values()
    ):
        raise ValueError(f"Missing paired validation games in {path}")
    return rows


def scores(rows: list[dict[str, str]], group_field: str, success_field: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        result[row[group_field]][row["map"]] += int(row[success_field])
    return {name: dict(by_map) for name, by_map in result.items()}


def comparison(candidate: dict[str, int], baseline: dict[str, int]) -> dict[str, int | bool]:
    candidate_total = sum(candidate.values())
    baseline_total = sum(baseline.values())
    maps_not_worse = sum(candidate[name] >= baseline[name] for name in MAPS)
    return {
        "candidate": candidate_total,
        "baseline": baseline_total,
        "difference": candidate_total - baseline_total,
        "maps_not_worse": maps_not_worse,
        "passed": candidate_total - baseline_total >= 100 and maps_not_worse >= 7,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and lock V7 validation selection")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--lock", action="store_true")
    return parser.parse_args()


def select(results_dir: Path, model_dir: Path, v3_model_dir: Path = Path("models")) -> dict[str, object]:
    hashes: dict[str, str] = {}
    seeker_results: dict[str, dict[str, dict[str, int | bool]]] = {}
    hider_results: dict[str, dict[str, dict[str, int | bool]]] = {}
    object_results: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for seed in SEEDS:
        seeker_results[str(seed)] = {}
        for hider in ("v3", "v7"):
            path = results_dir / f"v7_seeker_{hider}_seed{seed}.csv"
            rows = read_rows(
                path, len(MAPS) * EPISODES * 3,
                ("map", "training_seed", "eval_seed", "matchup"),
                3,
            )
            if {row["training_seed"] for row in rows} != {str(seed)}:
                raise ValueError(f"Training seed mismatch in {path}")
            by_policy = scores(rows, "matchup", "captured")
            if set(by_policy) != {"random", "teacher", "learned"}:
                raise ValueError(f"Seeker matchup mismatch in {path}")
            seeker_results[str(seed)][hider] = comparison(by_policy["learned"], by_policy["random"])
            hashes[path.name] = file_hash(path)

        path = results_dir / f"v7_hider_seed{seed}.csv"
        rows = read_rows(
            path, len(MAPS) * EPISODES * 6,
            ("map", "training_seed", "eval_seed", "seeker", "hider"),
            6,
        )
        if {row["training_seed"] for row in rows} != {str(seed)}:
            raise ValueError(f"Training seed mismatch in {path}")
        hider_results[str(seed)] = {}
        for seeker in ("random", "teacher"):
            subset = [row for row in rows if row["seeker"] == seeker]
            by_policy = scores(subset, "hider", "survived")
            if set(by_policy) != {"random", "v3", "v7"}:
                raise ValueError(f"Hider matchup mismatch in {path}")
            hider_results[str(seed)][seeker] = comparison(by_policy["v7"], by_policy["random"])
        hashes[path.name] = file_hash(path)

        path = results_dir / f"v7_objects_seed{seed}.csv"
        rows = read_rows(
            path, len(MAPS) * EPISODES * 8,
            ("map", "training_seed", "eval_seed", "role", "mode"),
            8,
        )
        if {row["training_seed"] for row in rows} != {str(seed)}:
            raise ValueError(f"Training seed mismatch in {path}")
        object_results[str(seed)] = {}
        for role in ("seeker", "hider"):
            object_results[str(seed)][role] = {}
            for mode in ("full", "no_block", "no_ramp", "no_objects"):
                subset = [row for row in rows if row["role"] == role and row["mode"] == mode]
                if len(subset) != len(MAPS) * EPISODES:
                    raise ValueError(f"Object mode mismatch in {path}")
                object_results[str(seed)][role][mode] = {
                    "success": sum(int(row["success"]) for row in subset),
                    "pushes": sum(int(row["pushes"]) for row in subset),
                    "crosses": sum(int(row["crosses"]) for row in subset),
                }
        hashes[path.name] = file_hash(path)

        for role, prefix in (("seeker", "seeker_v7_route"), ("hider", "hider_v7")):
            path = model_dir / f"{prefix}_seed{seed}.zip"
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if ((metadata.get("version"), metadata.get("role"), metadata.get("seed"))
                    != (7, role, seed)
                    or metadata.get("training_layout_hash") != layout_hash(TRAIN_LAYOUTS)):
                raise ValueError(f"V7 model metadata mismatch: {path}")
            expected = ({
                "route_hints": True, "teacher_steps": 20_000, "imitation_epochs": 5,
                "ppo_requested_steps": 25_000, "ppo_actual_steps": 25_088,
                "ppo_hider_pool": True, "observation_size": 516,
            } if role == "seeker" else {
                "ppo_requested_steps": 50_000, "ppo_actual_steps": 50_176,
                "opponent_pool": ["random", "observable_search_teacher"],
                "observation_size": 508,
            })
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"V7 training protocol mismatch: {path}")
            hashes[path.name] = file_hash(path)
            hashes[path.with_suffix(".json").name] = file_hash(path.with_suffix(".json"))

        v3_path = v3_model_dir / f"hider_v3_seed{seed}.zip"
        hashes[v3_path.name] = file_hash(v3_path)
        hashes[v3_path.with_suffix(".json").name] = file_hash(v3_path.with_suffix(".json"))

    seeker_pass = all(
        sum(bool(seeker_results[str(seed)][hider]["passed"]) for seed in SEEDS) >= 2
        for hider in ("v3", "v7")
    )
    hider_pass = all(
        sum(bool(hider_results[str(seed)][seeker]["passed"]) for seed in SEEDS) >= 2
        for seeker in ("random", "teacher")
    )
    object_by_seed = {
        str(seed): {
            "seeker_gain": (
                object_results[str(seed)]["seeker"]["full"]["success"]
                - object_results[str(seed)]["seeker"]["no_ramp"]["success"]
            ),
            "seeker_crosses": object_results[str(seed)]["seeker"]["full"]["crosses"],
            "hider_gain": (
                object_results[str(seed)]["hider"]["full"]["success"]
                - object_results[str(seed)]["hider"]["no_block"]["success"]
            ),
            "hider_pushes": object_results[str(seed)]["hider"]["full"]["pushes"],
        }
        for seed in SEEDS
    }
    object_pass = sum(
        value["seeker_gain"] >= 50 and value["seeker_crosses"] >= 20
        and value["hider_gain"] >= 50 and value["hider_pushes"] >= 20
        for value in object_by_seed.values()
    ) >= 2
    return {
        "version": 7,
        "validation_layout_hash": layout_hash(VALIDATION_LAYOUTS),
        "training_layout_hash": layout_hash(TRAIN_LAYOUTS),
        "episodes_per_map_per_seed": EPISODES,
        "seeker_results": seeker_results,
        "hider_results": hider_results,
        "object_results": object_results,
        "object_by_seed": object_by_seed,
        "seeker_pass": seeker_pass,
        "hider_pass": hider_pass,
        "object_pass": object_pass,
        "eligible_for_final": seeker_pass and hider_pass and object_pass,
        "artifact_hashes": hashes,
    }


def main() -> None:
    args = parse_args()
    result = select(args.results_dir, args.model_dir, args.v3_model_dir)
    print(json.dumps({
        "seeker_pass": result["seeker_pass"],
        "hider_pass": result["hider_pass"],
        "object_pass": result["object_pass"],
        "eligible_for_final": result["eligible_for_final"],
        "seeker_results": result["seeker_results"],
        "hider_results": result["hider_results"],
        "object_results": result["object_results"],
    }, indent=2))
    if args.lock:
        path = args.results_dir / "v7_selection.json"
        if path.exists():
            raise FileExistsError(f"Selection already locked: {path}")
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Locked: {path}")


if __name__ == "__main__":
    main()
