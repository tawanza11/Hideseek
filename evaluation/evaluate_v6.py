"""Evaluate the registered V6 experiment and lock selection before final maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.hide_seek_v3_env import DEFAULT_MAP
from env.v5_maps import VALIDATION_MAPS
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v31 import run_matchup
from evaluation.evaluate_v4_models import load_v4_model
from evaluation.evaluate_v5 import load_v5_model


MATCHUPS = ("Random", "V4 no memory", "V5 control", "V6 flat", "V6 curriculum")
EVAL_SEEDS = (41, 42, 43)
EPISODES_PER_MAP = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and lock the V6 comparison")
    parser.add_argument(
        "--partition", choices=("validation", "default", "lock", "final"),
        default="validation",
    )
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v4-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v5-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v6-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    return args


def load_v6_model(path: Path, *, variant: str, seed: int, max_steps: int) -> PPO:
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("version") != 6
        or data.get("role") != "seeker"
        or data.get("variant") != variant
        or data.get("seed") != seed
        or data.get("max_steps") != max_steps
        or data.get("action_mask_during_training") is not False
    ):
        raise ValueError(f"V6 model metadata mismatch: {path}")
    model = PPO.load(str(path), device="cpu")
    if model.observation_space.shape != (105,):
        raise ValueError(f"V6 model observation size mismatch: {path}")
    return model


def result_path(output_dir: Path, partition: str) -> Path:
    return output_dir / "results" / f"v6_{partition}_evaluation.csv"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_results(path: Path, *, maps: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    expected = maps * len(EVAL_SEEDS) * EPISODES_PER_MAP * len(MATCHUPS)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} evaluation rows in {path}; got {len(rows)}")
    combinations = {
        (row["map"], int(row["training_seed"]), int(row["eval_seed"]), row["matchup"])
        for row in rows
    }
    if len(combinations) != expected or {row["matchup"] for row in rows} != set(MATCHUPS):
        raise ValueError(f"Missing or duplicated evaluation games in {path}")
    return rows


def capture_count(rows: list[dict[str, str]], matchup: str, seed: int | None = None) -> int:
    return sum(
        int(row["captured"])
        for row in rows
        if row["matchup"] == matchup
        and (seed is None or int(row["training_seed"]) == seed)
    )


def lock_selection(output_dir: Path) -> dict[str, object]:
    selection_path = output_dir / "results" / "v6_selection.json"
    if selection_path.exists():
        raise FileExistsError(f"V6 selection is already locked: {selection_path}")
    validation_path = result_path(output_dir, "validation")
    default_path = result_path(output_dir, "default")
    validation = read_results(validation_path, maps=len(VALIDATION_MAPS))
    default = read_results(default_path, maps=1)
    validation_games = len(VALIDATION_MAPS) * len(EVAL_SEEDS) * EPISODES_PER_MAP
    default_games = len(EVAL_SEEDS) * EPISODES_PER_MAP
    curriculum = capture_count(validation, "V6 curriculum")
    flat = capture_count(validation, "V6 flat")
    v5 = capture_count(validation, "V5 control")
    seed_wins = sum(
        capture_count(validation, "V6 curriculum", seed)
        > capture_count(validation, "V6 flat", seed)
        for seed in EVAL_SEEDS
    )
    default_curriculum = capture_count(default, "V6 curriculum")
    default_v4 = capture_count(default, "V4 no memory")
    passed = (
        curriculum - flat >= 0.02 * validation_games
        and curriculum - v5 >= 0.02 * validation_games
        and seed_wins >= 2
        and default_curriculum >= default_v4 - 0.05 * default_games
    )
    decision: dict[str, object] = {
        "version": 6,
        "selection": "curriculum" if passed else None,
        "validation_sha256": file_hash(validation_path),
        "default_sha256": file_hash(default_path),
        "validation_games_per_matchup": validation_games,
        "default_games_per_matchup": default_games,
        "validation_captures": {
            "curriculum": curriculum, "flat": flat, "v5_control": v5,
        },
        "curriculum_seed_wins_over_flat": seed_wins,
        "default_captures": {
            "curriculum": default_curriculum, "v4_no_memory": default_v4,
        },
        "criteria": "validation curriculum >= flat +2pp and V5 +2pp; wins >=2/3 seeds; default >= V4 -5pp",
    }
    selection_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision


def require_lock(output_dir: Path) -> dict[str, object]:
    path = output_dir / "results" / "v6_selection.json"
    if not path.is_file():
        raise FileNotFoundError("Lock V6 validation and default results before opening final maps")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 6:
        raise ValueError(f"Invalid V6 selection file: {path}")
    for partition in ("validation", "default"):
        if data.get(f"{partition}_sha256") != file_hash(result_path(output_dir, partition)):
            raise ValueError(f"V6 {partition} results changed since selection lock")
    return data


def plot_results(rows: list[dict[str, int | str]], maps: tuple[str, ...], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(11, len(maps) * 1.3), 5))
    width = 0.16
    for index, name in enumerate(MATCHUPS):
        rates = [
            mean(int(row["captured"]) for row in rows if row["map"] == map_name and row["matchup"] == name)
            for map_name in maps
        ]
        axis.bar(
            [map_index + (index - 2) * width for map_index in range(len(maps))],
            rates, width=width, label=name,
        )
    axis.set_xticks(range(len(maps)), maps, rotation=40, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Capture rate against paired V3 hider")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> None:
    if args.partition == "validation":
        maps = VALIDATION_MAPS
        seed_start = 400_000
    elif args.partition == "default":
        maps = {"default": DEFAULT_MAP}
        seed_start = 500_000
    else:
        require_lock(args.output_dir)
        if result_path(args.output_dir, "final").exists():
            raise FileExistsError("V6 final results already exist; final maps may be evaluated once")
        from env.v5_maps import FINAL_TEST_MAPS
        maps = FINAL_TEST_MAPS
        seed_start = 800_000
    rows: list[dict[str, int | str]] = []
    for training_seed in EVAL_SEEDS:
        hider, max_steps = load_model(
            args.v3_model_dir / f"hider_v3_seed{training_seed}.zip",
            role="hider", stage="final", seed=training_seed,
        )
        v4 = load_v4_model(
            args.v4_model_dir / f"seeker_v4_nomemory_seed{training_seed}.zip",
            variant="nomemory", seed=training_seed, max_steps=max_steps,
        )
        v5 = load_v5_model(
            args.v5_model_dir / f"seeker_v5_control_seed{training_seed}.zip",
            variant="control", seed=training_seed, max_steps=max_steps,
        )
        if not isinstance(v5, PPO):
            raise TypeError("V5 control model is not PPO")
        flat = load_v6_model(
            args.v6_model_dir / f"seeker_v6_flat_seed{training_seed}.zip",
            variant="flat", seed=training_seed, max_steps=max_steps,
        )
        curriculum = load_v6_model(
            args.v6_model_dir / f"seeker_v6_curriculum_seed{training_seed}.zip",
            variant="curriculum", seed=training_seed, max_steps=max_steps,
        )
        seekers = (None, v4, v5, flat, curriculum)
        for map_index, (map_name, map_rows) in enumerate(maps.items()):
            for episode in range(EPISODES_PER_MAP):
                eval_seed = seed_start + map_index * 10_000 + episode
                for name, seeker in zip(MATCHUPS, seekers, strict=True):
                    result = run_matchup(
                        seeker, hider, max_steps=max_steps, seed=eval_seed,
                        legal_mask=seeker is not None, map_rows=map_rows,
                    )
                    rows.append({
                        "partition": args.partition,
                        "map": map_name,
                        "training_seed": training_seed,
                        "eval_seed": eval_seed,
                        "matchup": name,
                        **result,
                    })
    path = result_path(args.output_dir, args.partition)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "partition", "map", "training_seed", "eval_seed", "matchup", "captured",
            "steps", "wall_hits", "unique_cells", "visible_steps", "initial_visible",
        ), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot_path = args.output_dir / "results" / f"v6_{args.partition}_metrics.png"
    plot_results(rows, tuple(maps), plot_path)
    for name in MATCHUPS:
        group = [row for row in rows if row["matchup"] == name]
        captured = sum(int(row["captured"]) for row in group)
        print(f"{args.partition}, {name}: {captured}/{len(group)} ({captured / len(group):.1%})", flush=True)
    print(f"Evaluation: {path}\nChart: {plot_path}")


def main() -> None:
    args = parse_args()
    if args.partition == "lock":
        print(json.dumps(lock_selection(args.output_dir), indent=2))
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
