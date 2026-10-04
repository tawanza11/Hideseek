"""Compare V1/V2 seekers against random and trained hiders."""

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

from env.hide_seek_env import HideSeekEnv
from training.train_v2 import load_v1_config, model_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V2 Hide-and-Seek matchups")
    parser.add_argument("--v1-seeker", type=Path, default=Path("models/seeker_ppo.zip"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed-starts", type=int, nargs="+", default=[1000, 2000, 3000])
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--watch", action="store_true", help="Show one V2 seeker vs V2 hider episode")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()
    if args.episodes_per_seed <= 0 or args.fps <= 0:
        parser.error("--episodes-per-seed and --fps must be positive")
    if len(set(args.seed_starts)) != len(args.seed_starts):
        parser.error("--seed-starts must not repeat")
    return args


def check_model_metadata(path: Path, role: str, grid_size: int, max_steps: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Train V2 first: {path}")
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or (
        data.get("role") != role
        or data.get("grid_size") != grid_size
        or data.get("max_steps") != max_steps
    ):
        raise ValueError(f"Model metadata does not match the V1 game: {path}")


def run_matchup(
    seeker: PPO | None,
    hider: PPO | None,
    grid_size: int,
    max_steps: int,
    seed: int,
    watch: bool,
    fps: int,
) -> tuple[bool, int, float]:
    env = HideSeekEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        role="seeker",
        opponent_policy=model_policy(hider) if hider is not None else None,
        render_mode="human" if watch else None,
    )
    env.metadata = {**env.metadata, "render_fps": fps}
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    total_reward = 0.0
    try:
        for step in range(1, max_steps + 1):
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
                return bool(info["captured"]), step, total_reward
    finally:
        env.close()
    raise RuntimeError("Environment did not end at max_steps")


def save_plot(rows: list[dict[str, int | float | str]], path: Path) -> None:
    names = [
        "Random / random",
        "Random / V2 hider",
        "V1 / random",
        "V1 / V2 hider",
        "V2 / random",
        "V2 / V2 hider",
    ]
    labels = [
        "Random / random",
        "Random / V2 hider",
        "V1 / random",
        "V1 / V2 hider",
        "V2 / random",
        "V2 / V2 hider",
    ]
    seed_starts = sorted({int(row["seed_start"]) for row in rows})
    group_rates = [
        [
            mean(
                int(row["captured"])
                for row in rows
                if row["matchup"] == name and row["seed_start"] == seed_start
            )
            for seed_start in seed_starts
        ]
        for name in names
    ]
    step_groups = [
        [
            mean(
                int(row["steps"])
                for row in rows
                if row["matchup"] == name and row["seed_start"] == seed_start
            )
            for seed_start in seed_starts
        ]
        for name in names
    ]
    figure, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = ("#aaaaaa", "#db9b5a", "#6c9bd2", "#db9b5a", "#6c9bd2", "#3977b8")
    for axis, values, title, ylabel in (
        (axes[0], group_rates, "Seeker capture rate", "Capture rate"),
        (axes[1], step_groups, "Episode length", "Mean steps"),
    ):
        axis.bar(
            range(len(names)),
            [mean(group) for group in values],
            yerr=[pstdev(group) for group in values],
            capsize=5,
            color=colors,
        )
        axis.set_xticks(range(len(names)), labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    axes[0].set_ylim(0, 1.05)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    grid_size, max_steps = load_v1_config(args.v1_seeker)
    hider_path = args.model_dir / "hider_v2.zip"
    seeker_path = args.model_dir / "seeker_v2.zip"
    check_model_metadata(hider_path, "hider", grid_size, max_steps)
    check_model_metadata(seeker_path, "seeker", grid_size, max_steps)
    v1_seeker = PPO.load(str(args.v1_seeker), device="cpu")
    v2_seeker = PPO.load(str(seeker_path), device="cpu")
    v2_hider = PPO.load(str(hider_path), device="cpu")
    matchups = (
        ("Random / random", None, None),
        ("Random / V2 hider", None, v2_hider),
        ("V1 / random", v1_seeker, None),
        ("V1 / V2 hider", v1_seeker, v2_hider),
        ("V2 / random", v2_seeker, None),
        ("V2 / V2 hider", v2_seeker, v2_hider),
    )
    rows: list[dict[str, int | float | str]] = []
    for seed_start in args.seed_starts:
        for episode in range(args.episodes_per_seed):
            seed = seed_start + episode
            for name, seeker, hider in matchups:
                watch = args.watch and seed_start == args.seed_starts[0] and episode == 0 and name == "V2 / V2 hider"
                captured, steps, reward = run_matchup(
                    seeker, hider, grid_size, max_steps, seed, watch, args.fps
                )
                rows.append(
                    {
                        "matchup": name,
                        "seed_start": seed_start,
                        "episode": episode + 1,
                        "seed": seed,
                        "captured": int(captured),
                        "steps": steps,
                        "seeker_reward": round(reward, 4),
                    }
                )
    result_path = args.output_dir / "results/v2_evaluation.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("matchup", "seed_start", "episode", "seed", "captured", "steps", "seeker_reward"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    for name, _, _ in matchups:
        group = [row for row in rows if row["matchup"] == name]
        captures = sum(int(row["captured"]) for row in group)
        mean_steps = mean(int(row["steps"]) for row in group)
        print(
            f"{name}: {captures}/{len(group)} captures "
            f"({captures / len(group):.1%}), {mean_steps:.2f} mean steps"
        )
    plot_path = args.output_dir / "results/v2_metrics.png"
    save_plot(rows, plot_path)
    print(f"Evaluation: {result_path}")
    print(f"Chart: {plot_path}")


if __name__ == "__main__":
    main()
