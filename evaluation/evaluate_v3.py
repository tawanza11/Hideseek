"""Evaluate V3 on unseen episode seeds across independent training seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.hide_seek_v3_env import DEFAULT_MAP, HideSeekV3Env
from training.train_v2 import model_policy


MATCHUP_NAMES = (
    "Random / random",
    "Random / V3 hider",
    "Baseline seeker / random",
    "Baseline seeker / V3 hider",
    "Final seeker / random",
    "Final seeker / V3 hider",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V3 room-and-visibility agents")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--eval-seed-starts", type=int, nargs="+", default=[10000, 20000, 30000])
    parser.add_argument("--episodes-per-group", type=int, default=100)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--watch", action="store_true", help="Show one final seeker vs trained hider game")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()
    if args.episodes_per_group <= 0 or args.fps <= 0:
        parser.error("--episodes-per-group and --fps must be positive")
    if len(args.training_seeds) != len(set(args.training_seeds)):
        parser.error("--training-seeds must not repeat")
    if len(args.eval_seed_starts) != len(set(args.eval_seed_starts)):
        parser.error("--eval-seed-starts must not repeat")
    return args


def load_model(path: Path, *, role: str, stage: str, seed: int) -> tuple[PPO, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Train V3 first: {path}")
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or (
        data.get("version") != 3
        or data.get("role") != role
        or data.get("stage") != stage
        or data.get("seed") != seed
        or data.get("map_rows") != list(DEFAULT_MAP)
    ):
        raise ValueError(f"V3 model metadata mismatch: {path}")
    max_steps = data.get("max_steps")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"Invalid max_steps in model metadata: {path}")
    return PPO.load(str(path), device="cpu"), max_steps


def run_matchup(
    seeker: PPO | None,
    hider: PPO | None,
    *,
    max_steps: int,
    seed: int,
    watch: bool,
    fps: int,
) -> tuple[bool, int, int, float]:
    env = HideSeekV3Env(
        max_steps=max_steps,
        role="seeker",
        opponent_policy=model_policy(hider) if hider is not None else None,
        render_mode="human" if watch else None,
    )
    env.metadata = {**env.metadata, "render_fps": fps}
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    visible_steps = 0
    total_reward = 0.0
    try:
        for step in range(1, max_steps + 1):
            visible_steps += int(env.is_visible("seeker"))
            if watch:
                env.render()
            if seeker is None:
                action = env.action_space.sample()
            else:
                action, _ = seeker.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            if terminated or truncated:
                if watch:
                    env.render()
                return bool(info["captured"]), step, visible_steps, total_reward
    finally:
        env.close()
    raise RuntimeError("Environment did not end at max_steps")


def save_plot(rows: list[dict[str, int | float | str]], path: Path) -> None:
    training_seeds = sorted({int(row["training_seed"]) for row in rows})
    captures: list[list[float]] = []
    steps: list[list[float]] = []
    for name in MATCHUP_NAMES:
        captures.append(
            [
                mean(
                    int(row["captured"])
                    for row in rows
                    if row["matchup"] == name and row["training_seed"] == seed
                )
                for seed in training_seeds
            ]
        )
        steps.append(
            [
                mean(
                    int(row["steps"])
                    for row in rows
                    if row["matchup"] == name and row["training_seed"] == seed
                )
                for seed in training_seeds
            ]
        )
    figure, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = ("#aaaaaa", "#db9b5a", "#6c9bd2", "#db9b5a", "#6c9bd2", "#3977b8")
    for axis, values, title, ylabel in (
        (axes[0], captures, "Seeker capture rate", "Capture rate"),
        (axes[1], steps, "Episode length", "Mean steps"),
    ):
        axis.bar(
            range(len(MATCHUP_NAMES)),
            [mean(group) for group in values],
            yerr=[pstdev(group) for group in values],
            capsize=5,
            color=colors,
        )
        axis.set_xticks(range(len(MATCHUP_NAMES)), MATCHUP_NAMES, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    axes[0].set_ylim(0, 1.05)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, int | float | str]] = []
    for training_seed in args.training_seeds:
        baseline, baseline_steps = load_model(
            args.model_dir / f"seeker_v3_baseline_seed{training_seed}.zip",
            role="seeker",
            stage="baseline",
            seed=training_seed,
        )
        final_seeker, seeker_steps = load_model(
            args.model_dir / f"seeker_v3_seed{training_seed}.zip",
            role="seeker",
            stage="final",
            seed=training_seed,
        )
        final_hider, hider_steps = load_model(
            args.model_dir / f"hider_v3_seed{training_seed}.zip",
            role="hider",
            stage="final",
            seed=training_seed,
        )
        if len({baseline_steps, seeker_steps, hider_steps}) != 1:
            raise ValueError(f"V3 model time limits differ for seed {training_seed}")
        matchups = (
            (MATCHUP_NAMES[0], None, None),
            (MATCHUP_NAMES[1], None, final_hider),
            (MATCHUP_NAMES[2], baseline, None),
            (MATCHUP_NAMES[3], baseline, final_hider),
            (MATCHUP_NAMES[4], final_seeker, None),
            (MATCHUP_NAMES[5], final_seeker, final_hider),
        )
        for eval_seed_start in args.eval_seed_starts:
            for episode in range(args.episodes_per_group):
                eval_seed = eval_seed_start + episode
                for name, seeker, hider in matchups:
                    watch = (
                        args.watch
                        and training_seed == args.training_seeds[0]
                        and eval_seed_start == args.eval_seed_starts[0]
                        and episode == 0
                        and name == MATCHUP_NAMES[-1]
                    )
                    captured, steps, visible_steps, reward = run_matchup(
                        seeker,
                        hider,
                        max_steps=seeker_steps,
                        seed=eval_seed,
                        watch=watch,
                        fps=args.fps,
                    )
                    rows.append(
                        {
                            "training_seed": training_seed,
                            "eval_seed_start": eval_seed_start,
                            "episode": episode + 1,
                            "eval_seed": eval_seed,
                            "matchup": name,
                            "captured": int(captured),
                            "steps": steps,
                            "visible_steps": visible_steps,
                            "seeker_reward": round(reward, 4),
                        }
                    )
    result_path = args.output_dir / "results/v3_evaluation.csv"
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
                "visible_steps",
                "seeker_reward",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    for name in MATCHUP_NAMES:
        group = [row for row in rows if row["matchup"] == name]
        captures = sum(int(row["captured"]) for row in group)
        mean_steps = mean(int(row["steps"]) for row in group)
        visible_rate = sum(int(row["visible_steps"]) for row in group) / sum(
            int(row["steps"]) for row in group
        )
        print(
            f"{name}: {captures}/{len(group)} captures ({captures / len(group):.1%}), "
            f"{mean_steps:.2f} mean steps, visible {visible_rate:.1%} of steps"
        )
    plot_path = args.output_dir / "results/v3_metrics.png"
    save_plot(rows, plot_path)
    print(f"Evaluation: {result_path}")
    print(f"Chart: {plot_path}")


if __name__ == "__main__":
    main()
