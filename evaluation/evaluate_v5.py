"""Evaluate V5 action-mask training on unseen validation maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from env.hide_seek_v3_env import DEFAULT_MAP
from env.hide_seek_v5_env import HideSeekV5Env
from env.v4_maps import TEST_MAPS as V4_TEST_MAPS
from env.v5_maps import VALIDATION_MAPS
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v31 import load_v31_model, run_matchup
from evaluation.evaluate_v4_models import load_v4_model
from training.train_v2 import model_policy


MATCHUPS = ("Random", "V3.1 masked", "V4 no memory", "V5 control", "V5 masked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V5 seekers on held-out maps")
    parser.add_argument(
        "--partition", choices=("validation", "default", "legacy"), default="validation"
    )
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--episodes-per-map", type=int, default=100)
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v31-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v4-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v5-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.episodes_per_map <= 0:
        parser.error("--episodes-per-map must be positive")
    if not args.training_seeds or any(seed < 0 for seed in args.training_seeds):
        parser.error("--training-seeds must be nonempty and nonnegative")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("--training-seeds must be unique")
    return args


def load_v5_model(
    path: Path, *, variant: str, seed: int, max_steps: int
) -> PPO | MaskablePPO:
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("version") != 5
        or data.get("role") != "seeker"
        or data.get("variant") != variant
        or data.get("seed") != seed
        or data.get("max_steps") != max_steps
        or data.get("action_mask_during_training") != (variant == "masked")
    ):
        raise ValueError(f"V5 model metadata mismatch: {path}")
    model_class = MaskablePPO if variant == "masked" else PPO
    model = model_class.load(str(path), device="cpu")
    if model.observation_space.shape != (105,):
        raise ValueError(f"V5 model observation size mismatch: {path}")
    return model


def run_masked_matchup(
    seeker: MaskablePPO,
    hider: PPO,
    *,
    map_rows: tuple[str, ...],
    max_steps: int,
    seed: int,
) -> dict[str, int]:
    env = HideSeekV5Env(
        map_pool=(map_rows,),
        novelty_bonus=0,
        max_steps=max_steps,
        opponent_policy=model_policy(hider),
    )
    observation, _ = env.reset(seed=seed)
    initial_visible = int(env.is_visible("seeker"))
    visited = {env.seeker_pos}
    visible_steps = 0
    wall_hits = 0
    try:
        for step in range(1, max_steps + 1):
            visible_steps += int(env.is_visible("seeker"))
            action, _ = seeker.predict(
                observation, deterministic=True, action_masks=env.action_masks()
            )
            chosen = int(action)
            dx, dy = env._ACTION_DELTAS[chosen]
            attempted = (env.seeker_pos[0] + dx, env.seeker_pos[1] + dy)
            wall_hits += int(not env._inside(attempted))
            observation, _, terminated, truncated, info = env.step(chosen)
            visited.add(env.seeker_pos)
            if terminated or truncated:
                return {
                    "captured": int(info["captured"]),
                    "steps": step,
                    "wall_hits": wall_hits,
                    "unique_cells": len(visited),
                    "visible_steps": visible_steps,
                    "initial_visible": initial_visible,
                }
    finally:
        env.close()
    raise RuntimeError("Environment did not end at max_steps")


def save_plot(rows: list[dict[str, int | str]], maps: tuple[str, ...], path: Path) -> None:
    seeds = sorted({int(row["training_seed"]) for row in rows})
    figure, axis = plt.subplots(figsize=(max(11, len(maps) * 1.3), 5))
    width = 0.16
    for index, name in enumerate(MATCHUPS):
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
        axis.bar(
            [map_index + (index - 2) * width for map_index in range(len(maps))],
            [mean(group) for group in groups],
            width=width,
            yerr=[pstdev(group) for group in groups],
            capsize=2,
            label=name,
        )
    axis.set_xticks(range(len(maps)), maps, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Capture rate against trained V3 hider")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.partition == "validation":
        maps = VALIDATION_MAPS
        seed_start = 400_000
    elif args.partition == "default":
        maps = {"default": DEFAULT_MAP}
        seed_start = 500_000
    else:
        maps = V4_TEST_MAPS
        seed_start = 600_000
    rows: list[dict[str, int | str]] = []
    for training_seed in args.training_seeds:
        hider, max_steps = load_model(
            args.v3_model_dir / f"hider_v3_seed{training_seed}.zip",
            role="hider", stage="final", seed=training_seed,
        )
        v31 = load_v31_model(
            args.v31_model_dir / f"seeker_v31_control_seed{training_seed}.zip",
            variant="control", seed=training_seed, max_steps=max_steps,
        )
        v4 = load_v4_model(
            args.v4_model_dir / f"seeker_v4_nomemory_seed{training_seed}.zip",
            variant="nomemory", seed=training_seed, max_steps=max_steps,
        )
        control = load_v5_model(
            args.v5_model_dir / f"seeker_v5_control_seed{training_seed}.zip",
            variant="control", seed=training_seed, max_steps=max_steps,
        )
        masked = load_v5_model(
            args.v5_model_dir / f"seeker_v5_masked_seed{training_seed}.zip",
            variant="masked", seed=training_seed, max_steps=max_steps,
        )
        if not isinstance(control, PPO) or not isinstance(masked, MaskablePPO):
            raise TypeError("V5 model variants loaded with unexpected algorithm")
        for map_index, (map_name, map_rows) in enumerate(maps.items()):
            for episode in range(args.episodes_per_map):
                eval_seed = seed_start + map_index * 10_000 + episode
                for name, seeker in (
                    (MATCHUPS[0], None),
                    (MATCHUPS[1], v31),
                    (MATCHUPS[2], v4),
                    (MATCHUPS[3], control),
                    (MATCHUPS[4], masked),
                ):
                    result = (
                        run_masked_matchup(
                            masked, hider, map_rows=map_rows,
                            max_steps=max_steps, seed=eval_seed,
                        )
                        if name == MATCHUPS[4]
                        else run_matchup(
                            seeker, hider, max_steps=max_steps, seed=eval_seed,
                            legal_mask=seeker is not None, map_rows=map_rows,
                        )
                    )
                    rows.append({
                        "partition": args.partition,
                        "map": map_name,
                        "training_seed": training_seed,
                        "eval_seed": eval_seed,
                        "matchup": name,
                        **result,
                    })
    result_path = args.output_dir / "results" / f"v5_{args.partition}_evaluation.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "partition", "map", "training_seed", "eval_seed", "matchup", "captured",
            "steps", "wall_hits", "unique_cells", "visible_steps", "initial_visible",
        ), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for name in MATCHUPS:
        group = [row for row in rows if row["matchup"] == name]
        captures = sum(int(row["captured"]) for row in group)
        print(
            f"{args.partition}, {name}: {captures}/{len(group)} "
            f"({captures / len(group):.1%}), "
            f"{mean(int(row['unique_cells']) for row in group):.2f} unique cells",
            flush=True,
        )
    for map_name in maps:
        rates = []
        for name in MATCHUPS:
            group = [
                row for row in rows
                if row["map"] == map_name and row["matchup"] == name
            ]
            captures = sum(int(row["captured"]) for row in group)
            rates.append(f"{name} {captures}/{len(group)}")
        print(f"{map_name}: {', '.join(rates)}", flush=True)
    plot_path = args.output_dir / "results" / f"v5_{args.partition}_metrics.png"
    save_plot(rows, tuple(maps), plot_path)
    print(f"Evaluation: {result_path}\nChart: {plot_path}")


if __name__ == "__main__":
    main()
