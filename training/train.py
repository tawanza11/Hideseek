"""Train the seeker with PPO and record episode results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env.hide_seek_env import HideSeekEnv


class EpisodeLogger(BaseCallback):
    def __init__(self, output: Path) -> None:
        super().__init__()
        self.output = output
        self.episode = 0
        self.current_reward = 0.0
        self.current_length = 0

    def _on_training_start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file, lineterminator="\n").writerow(
                ["episode", "timesteps", "reward", "length", "success"]
            )

    def _on_step(self) -> bool:
        self.current_reward += float(self.locals["rewards"][0])
        self.current_length += 1
        if bool(self.locals["dones"][0]):
            info = self.locals["infos"][0]
            self.episode += 1
            with self.output.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file, lineterminator="\n").writerow(
                    [
                        self.episode,
                        self.num_timesteps,
                        round(self.current_reward, 4),
                        self.current_length,
                        int(info["success"]),
                    ]
                )
            self.current_reward = 0.0
            self.current_length = 0
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the V1 seeker with PPO")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--model-path", type=Path, default=Path("models/seeker_ppo.zip"))
    parser.add_argument("--log-path", type=Path, default=Path("results/train_episodes.csv"))
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.model_path.suffix != ".zip":
        parser.error("--model-path must end in .zip")
    return args


def main() -> None:
    args = parse_args()
    env = HideSeekEnv(grid_size=args.grid_size, max_steps=args.max_steps)
    env.reset(seed=args.seed)
    logger = EpisodeLogger(args.log_path)
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        n_steps=512,
        batch_size=64,
        verbose=0,
    )
    try:
        model.learn(total_timesteps=args.timesteps, callback=logger)
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.model_path))
        args.model_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "grid_size": args.grid_size,
                    "max_steps": args.max_steps,
                    "seed": args.seed,
                    "train_log": str(args.log_path),
                    "requested_timesteps": args.timesteps,
                    "actual_timesteps": model.num_timesteps,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()
    print(f"Model: {args.model_path}")
    print(f"Episodes: {logger.episode}; log: {args.log_path}")


if __name__ == "__main__":
    main()
