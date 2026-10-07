"""Lock V8 validation evidence and exact models before final-map evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from env.v8_maps import TRAIN_LAYOUTS, VALIDATION_LAYOUTS, layout_hash


SEEDS = (41, 42, 43)
MAPS = tuple(VALIDATION_LAYOUTS)
EPISODES = 100
VARIANTS = {
    ("seeker", "full", "learned"), ("seeker", "full", "random"),
    ("seeker", "no_ramp", "learned"), ("seeker", "no_block", "learned"),
    ("seeker", "no_objects", "learned"),
    ("hider", "full", "learned"), ("hider", "full", "random"),
    ("hider", "no_block", "learned"), ("hider", "no_ramp", "learned"),
    ("hider", "no_objects", "learned"),
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(candidate: dict[str, int], baseline: dict[str, int]) -> dict[str, int | bool]:
    gain = sum(candidate.values()) - sum(baseline.values())
    maps_not_worse = sum(candidate[name] >= baseline[name] for name in MAPS)
    return {
        "candidate": sum(candidate.values()), "baseline": sum(baseline.values()),
        "gain": gain, "maps_not_worse": maps_not_worse,
        "passed": gain >= 100 and maps_not_worse >= 7,
    }


def select(results_dir: Path, model_dir: Path, v7_model_dir: Path) -> dict[str, object]:
    hashes: dict[str, str] = {}
    scores: dict[str, dict[str, dict[str, int | bool]]] = {}
    events: dict[str, dict[str, int]] = {}
    for seed in SEEDS:
        path = results_dir / f"v8_validation_seed{seed}.csv"
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != len(MAPS) * EPISODES * len(VARIANTS):
            raise ValueError(f"V8 validation row count mismatch: {path}")
        identities = [
            (row["map"], row["training_seed"], row["eval_seed"],
             row["role"], row["mode"], row["policy"])
            for row in rows
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate V8 game: {path}")
        groups: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
        by_variant: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        event_counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for row in rows:
            if row["map"] not in MAPS or row["training_seed"] != str(seed):
                raise ValueError(f"V8 map or training seed mismatch: {path}")
            variant = (row["role"], row["mode"], row["policy"])
            if variant not in VARIANTS:
                raise ValueError(f"V8 variant mismatch: {path}")
            groups[(row["map"], row["eval_seed"])].add(variant)
            success = int(row["success"])
            steps = int(row["steps"])
            pushes = int(row["pushes"])
            crosses = int(row["crosses"])
            if success not in (0, 1) or not 1 <= steps <= 27 or min(pushes, crosses) < 0:
                raise ValueError(f"Invalid V8 result value: {path}")
            by_variant[variant][row["map"]] += success
            event_counts[variant]["pushes"] += pushes
            event_counts[variant]["crosses"] += crosses
        if len(groups) != len(MAPS) * EPISODES or any(
            variants != VARIANTS for variants in groups.values()
        ):
            raise ValueError(f"Missing paired V8 games: {path}")
        scores[str(seed)] = {}
        events[str(seed)] = {}
        for role in ("seeker", "hider"):
            full = by_variant[(role, "full", "learned")]
            random = by_variant[(role, "full", "random")]
            ablated = by_variant[(role, "no_ramp" if role == "seeker" else "no_block", "learned")]
            scores[str(seed)][role] = comparison(full, random)
            events[str(seed)][role + "_gain"] = sum(full.values()) - sum(ablated.values())
            event_name = "crosses" if role == "seeker" else "pushes"
            events[str(seed)][role + "_events"] = event_counts[(role, "full", "learned")][event_name]
        hashes[path.name] = file_hash(path)

        for role in ("seeker", "hider"):
            model_path = model_dir / f"{role}_v8_seed{seed}.zip"
            metadata_path = model_path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_path = v7_model_dir / (
                f"seeker_v7_route_seed{seed}.zip" if role == "seeker"
                else f"hider_v7_seed{seed}.zip"
            )
            expected = {
                "version": 8, "role": role, "seed": seed,
                "source_model": source_path.name,
                "source_hash": file_hash(source_path),
                "training_layout_hash": layout_hash(TRAIN_LAYOUTS),
                "max_steps": 27, "imitation_steps": 20_000,
                "imitation_epochs": 5, "ppo_requested_steps": 100_000,
                "ppo_actual_steps": 100_352,
                "observation_size": 516 if role == "seeker" else 508,
                "object_reward": 0.1 if role == "seeker" else 0.3,
                "opponent": "barricade_demonstration" if role == "seeker"
                else "random_or_observable_search",
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"V8 training metadata mismatch: {model_path}")
            for artifact in (model_path, metadata_path, source_path,
                             source_path.with_suffix(".json")):
                hashes[artifact.name] = file_hash(artifact)

    basic_pass = all(
        sum(bool(scores[str(seed)][role]["passed"]) for seed in SEEDS) >= 2
        for role in ("seeker", "hider")
    )
    object_pass = all(
        sum(
            events[str(seed)][role + "_gain"] >= 100
            and events[str(seed)][role + "_events"] >= 20
            for seed in SEEDS
        ) >= 2
        for role in ("seeker", "hider")
    )
    return {
        "version": 8,
        "training_layout_hash": layout_hash(TRAIN_LAYOUTS),
        "validation_layout_hash": layout_hash(VALIDATION_LAYOUTS),
        "episodes_per_map_per_seed": EPISODES,
        "basic_scores": scores,
        "object_effects": events,
        "basic_pass": basic_pass,
        "object_pass": object_pass,
        "eligible_for_final": basic_pass and object_pass,
        "artifact_hashes": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock V8 validation selection")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v7-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--lock", action="store_true")
    args = parser.parse_args()
    result = select(args.results_dir, args.model_dir, args.v7_model_dir)
    print(json.dumps({key: result[key] for key in (
        "basic_scores", "object_effects", "basic_pass", "object_pass", "eligible_for_final"
    )}, indent=2))
    if args.lock:
        path = args.results_dir / "v8_selection.json"
        if path.exists():
            raise FileExistsError(f"V8 selection already locked: {path}")
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Locked: {path}")


if __name__ == "__main__":
    main()
