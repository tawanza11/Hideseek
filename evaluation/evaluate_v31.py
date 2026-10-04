"""Compare V3.1 wall-penalty and control seekers on unchanged V3 rules."""

from __future__ import annotations

import argparse
import csv
import json
from math import isqrt
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from env.hide_seek_v3_env import DEFAULT_MAP, HideSeekV3Env
from evaluation.evaluate_v3 import load_model
from training.train_v2 import model_policy


MATCHUPS = (
    "Random / V3 hider",
    "V3 seeker / V3 hider",
    "Control seeker / V3 hider",
    "Wall-penalty seeker / V3 hider",
    "V3 seeker / random",
    "Control seeker / random",
    "Wall-penalty seeker / random",
)
MASKED_MATCHUPS = (
    "Masked V3 seeker / V3 hider",
    "Masked control seeker / V3 hider",
    "Masked control seeker / random",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V3.1 wall-penalty experiment")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--eval-seed-starts", type=int, nargs="+", default=[40000, 50000, 60000])
    parser.add_argument("--episodes-per-group", type=int, default=100)
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v31-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--legal-mask", action="store_true", help="Also evaluate legal-action masking")
    args = parser.parse_args()
    if args.episodes_per_group <= 0:
        parser.error("--episodes-per-group must be positive")
    if any(seed < 0 for seed in args.training_seeds) or len(set(args.training_seeds)) != len(
        args.training_seeds
    ):
        parser.error("--training-seeds must be nonnegative and unique")
    if any(seed < 0 for seed in args.eval_seed_starts) or len(set(args.eval_seed_starts)) != len(
        args.eval_seed_starts
    ):
        parser.error("--eval-seed-starts must be nonnegative and unique")
    return args


def load_v31_model(path: Path, *, seed: int, variant: str, max_steps: int) -> PPO:
    if not path.is_file():
        raise FileNotFoundError(f"Train V3.1 first: {path}")
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or (
        data.get("version") != "3.1"
        or data.get("role") != "seeker"
        or data.get("variant") != variant
        or data.get("seed") != seed
        or data.get("map_rows") != list(DEFAULT_MAP)
        or data.get("max_steps") != max_steps
    ):
        raise ValueError(f"V3.1 model metadata mismatch: {path}")
    return PPO.load(str(path), device="cpu")


def legal_actions_from_observation(observation: np.ndarray) -> tuple[int, ...]:
    """Use only the Seeker's position and wall map already present in V3 input."""
    if observation.ndim != 1 or observation.size < 9:
        raise ValueError("Expected a flat V3 observation")
    grid_size = isqrt(observation.size - 5)
    if grid_size < 2 or grid_size * grid_size != observation.size - 5:
        raise ValueError("V3 observation does not contain a square wall map")
    x = round(float(observation[0]) * (grid_size - 1))
    y = round(float(observation[1]) * (grid_size - 1))
    walls = observation[5:]
    actions = []
    for index, (dx, dy) in enumerate(HideSeekV3Env._ACTION_DELTAS):
        target_x, target_y = x + dx, y + dy
        if (
            0 <= target_x < grid_size
            and 0 <= target_y < grid_size
            and walls[target_y * grid_size + target_x] < 0.5
        ):
            actions.append(index)
    if not actions:
        raise RuntimeError("Seeker has no legal actions")
    return tuple(actions)


def masked_action(model: PPO, observation: np.ndarray) -> int:
    """Select the highest-probability legal action using the V3 observation."""
    legal_actions = legal_actions_from_observation(observation)
    tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        probabilities = model.policy.get_distribution(tensor).distribution.probs[0]
    return max(legal_actions, key=lambda index: float(probabilities[index]))


def run_matchup(
    seeker: PPO | None,
    hider: PPO | None,
    *,
    max_steps: int,
    seed: int,
    legal_mask: bool = False,
) -> dict[str, int]:
    env = HideSeekV3Env(
        role="seeker",
        max_steps=max_steps,
        opponent_policy=model_policy(hider) if hider is not None else None,
    )
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    initial_visible = int(env.is_visible("seeker"))
    visited = {env.seeker_pos}
    wall_hits = 0
    visible_steps = 0
    try:
        for step in range(1, max_steps + 1):
            visible_steps += int(env.is_visible("seeker"))
            if seeker is None:
                action = int(env.action_space.sample())
            elif legal_mask:
                action = masked_action(seeker, observation)
            else:
                predicted, _ = seeker.predict(observation, deterministic=True)
                action = int(predicted)
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


def save_plot(
    rows: list[dict[str, int | str]],
    path: Path,
    names: tuple[str, ...],
    labels: tuple[str, ...],
) -> None:
    seeds = sorted({int(row["training_seed"]) for row in rows})
    captures = [
        [
            mean(
                int(row["captured"])
                for row in rows
                if row["matchup"] == name and row["training_seed"] == seed
            )
            for seed in seeds
        ]
        for name in names
    ]
    wall_rates = [
        [
            sum(
                int(row["wall_hits"])
                for row in rows
                if row["matchup"] == name and row["training_seed"] == seed
            )
            / sum(
                int(row["steps"])
                for row in rows
                if row["matchup"] == name and row["training_seed"] == seed
            )
            for seed in seeds
        ]
        for name in names
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ("#aaaaaa", "#6c9bd2", "#86a987", "#3977b8", "#d98b62")
    for axis, values, title in (
        (axes[0], captures, "Capture rate vs V3 hider"),
        (axes[1], wall_rates, "Blocked-move rate vs V3 hider"),
    ):
        axis.bar(
            range(len(names)),
            [mean(group) for group in values],
            yerr=[pstdev(group) for group in values],
            capsize=5,
            color=colors[: len(names)],
        )
        axis.set_xticks(range(len(names)), labels)
        axis.set_ylim(0, 1)
        axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, int | str]] = []
    for training_seed in args.training_seeds:
        v3_seeker, max_steps = load_model(
            args.v3_model_dir / f"seeker_v3_seed{training_seed}.zip",
            role="seeker",
            stage="final",
            seed=training_seed,
        )
        v3_hider, hider_steps = load_model(
            args.v3_model_dir / f"hider_v3_seed{training_seed}.zip",
            role="hider",
            stage="final",
            seed=training_seed,
        )
        if max_steps != hider_steps:
            raise ValueError(f"V3 model time limits differ for seed {training_seed}")
        control = load_v31_model(
            args.v31_model_dir / f"seeker_v31_control_seed{training_seed}.zip",
            seed=training_seed,
            variant="control",
            max_steps=max_steps,
        )
        penalty = load_v31_model(
            args.v31_model_dir / f"seeker_v31_penalty_seed{training_seed}.zip",
            seed=training_seed,
            variant="penalty",
            max_steps=max_steps,
        )
        matchups = [
            (MATCHUPS[0], None, v3_hider, False),
            (MATCHUPS[1], v3_seeker, v3_hider, False),
            (MATCHUPS[2], control, v3_hider, False),
            (MATCHUPS[3], penalty, v3_hider, False),
            (MATCHUPS[4], v3_seeker, None, False),
            (MATCHUPS[5], control, None, False),
            (MATCHUPS[6], penalty, None, False),
        ]
        if args.legal_mask:
            matchups.extend(
                (
                    (MASKED_MATCHUPS[0], v3_seeker, v3_hider, True),
                    (MASKED_MATCHUPS[1], control, v3_hider, True),
                    (MASKED_MATCHUPS[2], control, None, True),
                )
            )
        for eval_seed_start in args.eval_seed_starts:
            for episode in range(args.episodes_per_group):
                eval_seed = eval_seed_start + episode
                for name, seeker, hider, legal_mask in matchups:
                    result = run_matchup(
                        seeker,
                        hider,
                        max_steps=max_steps,
                        seed=eval_seed,
                        legal_mask=legal_mask,
                    )
                    rows.append(
                        {
                            "training_seed": training_seed,
                            "eval_seed_start": eval_seed_start,
                            "episode": episode + 1,
                            "eval_seed": eval_seed,
                            "matchup": name,
                            **result,
                        }
                    )
    suffix = "masked" if args.legal_mask else "reward"
    result_path = args.output_dir / f"results/v31_{suffix}_evaluation.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "training_seed",
                "eval_seed_start",
                "episode",
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
    for name in (*MATCHUPS, *(MASKED_MATCHUPS if args.legal_mask else ())):
        group = [row for row in rows if row["matchup"] == name]
        captures = sum(int(row["captured"]) for row in group)
        wall_hits = sum(int(row["wall_hits"]) for row in group)
        steps = sum(int(row["steps"]) for row in group)
        print(
            f"{name}: captures {captures}/{len(group)} ({captures / len(group):.1%}), "
            f"blocked moves {wall_hits / steps:.1%}, "
            f"unique cells {mean(int(row['unique_cells']) for row in group):.2f}"
        )
    plot_path = args.output_dir / f"results/v31_{suffix}_metrics.png"
    if args.legal_mask:
        plot_names = (MATCHUPS[0], MATCHUPS[1], MATCHUPS[2], *MASKED_MATCHUPS[:2])
        plot_labels = ("Random", "V3", "Control", "Masked V3", "Masked control")
    else:
        plot_names = MATCHUPS[:4]
        plot_labels = ("Random", "V3", "Control", "Wall penalty")
    save_plot(rows, plot_path, plot_names, plot_labels)
    print(f"Evaluation: {result_path}")
    print(f"Chart: {plot_path}")


if __name__ == "__main__":
    main()
