"""Compare V4 memory and no-memory seekers on held-out room maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO

from env.hide_seek_v3_env import DEFAULT_MAP
from env.hide_seek_v4_env import HideSeekV4Env
from env.v4_maps import TEST_MAPS, VALIDATION_MAPS
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v31 import legal_actions_from_observation, load_v31_model, run_matchup
from training.train_v2 import model_policy


MATCHUPS = (
    "Random",
    "V3.1 masked",
    "V4 no memory",
    "V4 memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V4 seekers on unseen maps")
    parser.add_argument(
        "--partition", choices=("validation", "test", "default"), default="validation"
    )
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--episodes-per-map", type=int, default=100)
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v31-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v4-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.episodes_per_map <= 0:
        parser.error("--episodes-per-map must be positive")
    if not args.training_seeds or any(seed < 0 for seed in args.training_seeds):
        parser.error("--training-seeds must be nonempty and nonnegative")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("--training-seeds must be unique")
    return args


def load_v4_model(path: Path, *, variant: str, seed: int, max_steps: int) -> PPO:
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("version") != 4
        or data.get("role") != "seeker"
        or data.get("variant") != variant
        or data.get("seed") != seed
        or data.get("max_steps") != max_steps
        or data.get("visit_memory") != (variant == "memory")
    ):
        raise ValueError(f"V4 model metadata mismatch: {path}")
    model = PPO.load(str(path), device="cpu")
    expected = 205 if variant == "memory" else 105
    if model.observation_space.shape != (expected,):
        raise ValueError(f"V4 model observation size mismatch: {path}")
    return model


def run_v4_matchup(
    seeker: PPO,
    hider: PPO,
    *,
    map_rows: tuple[str, ...],
    visit_memory: bool,
    max_steps: int,
    seed: int,
) -> dict[str, int]:
    env = HideSeekV4Env(
        map_pool=(map_rows,),
        visit_memory=visit_memory,
        novelty_bonus=0,
        max_steps=max_steps,
        opponent_policy=model_policy(hider),
    )
    observation, _ = env.reset(seed=seed)
    initial_visible = int(env.is_visible("seeker"))
    visited = {env.seeker_pos}
    wall_hits = 0
    visible_steps = 0
    try:
        for step in range(1, max_steps + 1):
            visible_steps += int(env.is_visible("seeker"))
            base_observation = observation[: 5 + env.grid_size * env.grid_size]
            legal_actions = legal_actions_from_observation(base_observation)
            tensor, _ = seeker.policy.obs_to_tensor(observation)
            with torch.no_grad():
                probabilities = seeker.policy.get_distribution(tensor).distribution.probs[0]
            action = max(legal_actions, key=lambda index: float(probabilities[index]))
            dx, dy = env._ACTION_DELTAS[action]
            attempted = (env.seeker_pos[0] + dx, env.seeker_pos[1] + dy)
            wall_hits += int(not env._inside(attempted))
            observation, _, terminated, truncated, info = env.step(action)
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
    figure, axis = plt.subplots(figsize=(12, 5))
    width = 0.2
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
            [map_index + (index - 1.5) * width for map_index in range(len(maps))],
            [mean(group) for group in groups],
            width=width,
            yerr=[pstdev(group) for group in groups],
            capsize=3,
            label=name,
        )
    axis.set_xticks(range(len(maps)), maps)
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
        seed_start = 100_000
    elif args.partition == "test":
        maps = TEST_MAPS
        seed_start = 200_000
    else:
        maps = {"default": DEFAULT_MAP}
        seed_start = 300_000
    rows: list[dict[str, int | str]] = []
    for training_seed in args.training_seeds:
        hider, max_steps = load_model(
            args.v3_model_dir / f"hider_v3_seed{training_seed}.zip",
            role="hider", stage="final", seed=training_seed,
        )
        control = load_v31_model(
            args.v31_model_dir / f"seeker_v31_control_seed{training_seed}.zip",
            variant="control", seed=training_seed, max_steps=max_steps,
        )
        no_memory = load_v4_model(
            args.v4_model_dir / f"seeker_v4_nomemory_seed{training_seed}.zip",
            variant="nomemory", seed=training_seed, max_steps=max_steps,
        )
        memory = load_v4_model(
            args.v4_model_dir / f"seeker_v4_memory_seed{training_seed}.zip",
            variant="memory", seed=training_seed, max_steps=max_steps,
        )
        for map_index, (map_name, map_rows) in enumerate(maps.items()):
            for episode in range(args.episodes_per_map):
                eval_seed = seed_start + map_index * 10_000 + episode
                for name, seeker, visit_memory in (
                    (MATCHUPS[0], None, False),
                    (MATCHUPS[1], control, False),
                    (MATCHUPS[2], no_memory, False),
                    (MATCHUPS[3], memory, True),
                ):
                    if seeker is None:
                        result = run_matchup(
                            None, hider, max_steps=max_steps, seed=eval_seed, map_rows=map_rows
                        )
                    elif name == MATCHUPS[1]:
                        result = run_matchup(
                            control, hider, max_steps=max_steps, seed=eval_seed,
                            legal_mask=True, map_rows=map_rows,
                        )
                    else:
                        result = run_v4_matchup(
                            seeker, hider, map_rows=map_rows,
                            visit_memory=visit_memory, max_steps=max_steps, seed=eval_seed,
                        )
                    rows.append({
                        "partition": args.partition,
                        "map": map_name,
                        "training_seed": training_seed,
                        "eval_seed": eval_seed,
                        "matchup": name,
                        **result,
                    })
    result_path = args.output_dir / "results" / f"v4_models_{args.partition}_evaluation.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "partition", "map", "training_seed", "eval_seed", "matchup", "captured", "steps",
            "wall_hits", "unique_cells", "visible_steps", "initial_visible",
        ), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for map_name in maps:
        for name in MATCHUPS:
            group = [row for row in rows if row["map"] == map_name and row["matchup"] == name]
            captures = sum(int(row["captured"]) for row in group)
            print(f"{map_name}, {name}: {captures}/{len(group)} ({captures / len(group):.1%}), "
                  f"{mean(int(row['unique_cells']) for row in group):.2f} unique cells", flush=True)
    plot_path = args.output_dir / "results" / f"v4_models_{args.partition}_metrics.png"
    save_plot(rows, tuple(maps), plot_path)
    print(f"Evaluation: {result_path}\nChart: {plot_path}")


if __name__ == "__main__":
    main()
