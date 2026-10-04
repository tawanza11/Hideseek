"""Compare a trained seeker against a random seeker on seeded episodes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.hide_seek_env import HideSeekEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and watch the V1 seeker")
    parser.add_argument("--model-path", type=Path, default=Path("models/seeker_ppo.zip"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--watch", action="store_true", help="Show the first trained episode")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--results-path", type=Path, default=Path("results/evaluation.csv"))
    parser.add_argument("--plot-path", type=Path, default=Path("results/metrics.png"))
    args = parser.parse_args()
    if args.episodes <= 0 or args.fps <= 0:
        parser.error("--episodes and --fps must be positive")
    if args.model_path.suffix != ".zip":
        parser.error("--model-path must end in .zip")
    return args


def load_env_config(model_path: Path) -> tuple[int, int, Path]:
    metadata_path = model_path.with_suffix(".json")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid model metadata: {metadata_path}")
    grid_size, max_steps = data.get("grid_size"), data.get("max_steps")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError(f"Invalid grid_size in {metadata_path}")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"Invalid max_steps in {metadata_path}")
    train_log = data.get("train_log")
    if not isinstance(train_log, str) or not train_log:
        raise ValueError(f"Invalid train_log in {metadata_path}")
    return grid_size, max_steps, Path(train_log)


def run_episode(
    env: HideSeekEnv,
    model: PPO | None,
    seed: int,
    watch: bool = False,
) -> tuple[bool, int, float]:
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    total_reward = 0.0
    for step in range(1, env.max_steps + 1):
        if watch:
            env.render()
        if model is None:
            action = env.action_space.sample()
        else:
            prediction, _ = model.predict(observation, deterministic=True)
            action = int(prediction)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            if watch:
                env.render()
            return bool(info["success"]), step, total_reward
    raise RuntimeError("Environment did not end at max_steps")


def save_plot(
    results: list[dict[str, int | float | str]], plot_path: Path, train_log: Path
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    with train_log.open(newline="", encoding="utf-8") as file:
        training = list(csv.DictReader(file))
    if training:
        window = min(50, len(training))
        successes = [int(row["success"]) for row in training]
        rolling = [
            sum(successes[max(0, index - window + 1) : index + 1])
            / min(index + 1, window)
            for index in range(len(successes))
        ]
        axes[0].plot(range(1, len(rolling) + 1), rolling)
        axes[0].set_title(f"Training success (last {window} episodes)")
    else:
        axes[0].set_title("No completed training episodes")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylim(0, 1.05)
    names = ("PPO", "Random")
    rates = [
        sum(int(row["success"]) for row in results if row["policy"] == name)
        / sum(row["policy"] == name for row in results)
        for name in names
    ]
    axes[1].bar(names, rates, color=("#3377bb", "#aaaaaa"))
    axes[1].set_title("Evaluation success rate")
    axes[1].set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Train a model first: {args.model_path}")
    grid_size, max_steps, train_log = load_env_config(args.model_path)
    model = PPO.load(str(args.model_path))
    env = HideSeekEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        render_mode="human" if args.watch else None,
    )
    env.metadata = {**env.metadata, "render_fps": args.fps}
    results: list[dict[str, int | float | str]] = []
    try:
        for episode in range(args.episodes):
            for policy, agent in (("PPO", model), ("Random", None)):
                success, steps, reward = run_episode(
                    env, agent, args.seed + episode, episode == 0 and policy == "PPO" and args.watch
                )
                if episode == 0 and policy == "PPO" and args.watch:
                    env.close()
                    env.render_mode = None
                results.append(
                    {
                        "policy": policy,
                        "episode": episode + 1,
                        "seed": args.seed + episode,
                        "success": int(success),
                        "steps": steps,
                        "reward": round(reward, 4),
                    }
                )
    finally:
        env.close()
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("policy", "episode", "seed", "success", "steps", "reward"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    for policy in ("PPO", "Random"):
        subset = [row for row in results if row["policy"] == policy]
        wins = sum(int(row["success"]) for row in subset)
        print(f"{policy}: {wins}/{len(subset)} captures ({wins / len(subset):.1%})")
    if train_log.is_file():
        save_plot(results, args.plot_path, train_log)
        print(f"Chart: {args.plot_path}")
    print(f"Evaluation: {args.results_path}")


if __name__ == "__main__":
    main()
