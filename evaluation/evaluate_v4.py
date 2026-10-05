"""Evaluate the saved V3.1 Seeker on unseen V4 maps before retraining."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.v4_maps import TEST_MAPS, VALIDATION_MAPS
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v31 import load_v31_model, run_matchup


MATCHUPS = (
    "Random / random",
    "V3.1 masked / random",
    "Random / V3 hider",
    "V3.1 masked / V3 hider",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V3.1 on unseen V4 maps")
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--episodes-per-map", type=int, default=100)
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v31-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.episodes_per_map <= 0:
        parser.error("--episodes-per-map must be positive")
    if any(seed < 0 for seed in args.training_seeds) or len(set(args.training_seeds)) != len(
        args.training_seeds
    ):
        parser.error("--training-seeds must be nonnegative and unique")
    return args


def save_plot(rows: list[dict[str, int | str]], maps: tuple[str, ...], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    seeds = sorted({int(row["training_seed"]) for row in rows})
    for axis, names, title in (
        (axes[0], MATCHUPS[:2], "Against random hider"),
        (axes[1], MATCHUPS[2:], "Against V3 hider"),
    ):
        for offset, name in enumerate(names):
            groups = [
                [
                    mean(
                        int(row["captured"])
                        for row in rows
                        if row["map"] == map_name
                        and row["matchup"] == name
                        and row["training_seed"] == seed
                    )
                    for seed in seeds
                ]
                for map_name in maps
            ]
            positions = [index + (offset - 0.5) * 0.36 for index in range(len(maps))]
            axis.bar(
                positions,
                [mean(group) for group in groups],
                width=0.36,
                yerr=[pstdev(group) for group in groups],
                capsize=4,
                label="Random seeker" if offset == 0 else "V3.1 masked seeker",
            )
        axis.set_xticks(range(len(maps)), maps)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Capture rate")
        axis.set_title(title)
        axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    maps = VALIDATION_MAPS if args.partition == "validation" else TEST_MAPS
    seed_start = 100_000 if args.partition == "validation" else 200_000
    rows: list[dict[str, int | str]] = []
    for training_seed in args.training_seeds:
        hider, max_steps = load_model(
            args.v3_model_dir / f"hider_v3_seed{training_seed}.zip",
            role="hider",
            stage="final",
            seed=training_seed,
        )
        seeker: PPO = load_v31_model(
            args.v31_model_dir / f"seeker_v31_control_seed{training_seed}.zip",
            seed=training_seed,
            variant="control",
            max_steps=max_steps,
        )
        matchups = (
            (MATCHUPS[0], None, None, False),
            (MATCHUPS[1], seeker, None, True),
            (MATCHUPS[2], None, hider, False),
            (MATCHUPS[3], seeker, hider, True),
        )
        for map_index, (map_name, map_rows) in enumerate(maps.items()):
            for episode in range(args.episodes_per_map):
                eval_seed = seed_start + map_index * 10_000 + episode
                for name, seeker_model, hider_model, legal_mask in matchups:
                    result = run_matchup(
                        seeker_model,
                        hider_model,
                        max_steps=max_steps,
                        seed=eval_seed,
                        legal_mask=legal_mask,
                        map_rows=map_rows,
                    )
                    rows.append(
                        {
                            "partition": args.partition,
                            "map": map_name,
                            "training_seed": training_seed,
                            "eval_seed": eval_seed,
                            "matchup": name,
                            **result,
                        }
                    )
    result_path = args.output_dir / f"results/v4_{args.partition}_evaluation.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "partition",
                "map",
                "training_seed",
                "eval_seed",
                "matchup",
                "captured",
                "steps",
                "wall_hits",
                "unique_cells",
                "visible_steps",
                "initial_visible",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    for map_name in maps:
        for name in MATCHUPS:
            group = [row for row in rows if row["map"] == map_name and row["matchup"] == name]
            captures = sum(int(row["captured"]) for row in group)
            print(
                f"{map_name}, {name}: {captures}/{len(group)} captures "
                f"({captures / len(group):.1%}), "
                f"{mean(int(row['unique_cells']) for row in group):.2f} unique cells"
            )
    plot_path = args.output_dir / f"results/v4_{args.partition}_metrics.png"
    save_plot(rows, tuple(maps), plot_path)
    print(f"Evaluation: {result_path}")
    print(f"Chart: {plot_path}")


if __name__ == "__main__":
    main()
