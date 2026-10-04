"""Alternate PPO training for a hider and seeker, freezing each opponent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO

from env.hide_seek_env import HideSeekEnv
from training.train import EpisodeLogger


def model_policy(model: PPO) -> Callable[[np.ndarray], int]:
    def act(observation: np.ndarray) -> int:
        action, _ = model.predict(observation, deterministic=True)
        return int(action)

    return act


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train both Hide-and-Seek agents")
    parser.add_argument("--v1-seeker", type=Path, default=Path("models/seeker_ppo.zip"))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timesteps-per-phase", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.rounds <= 0 or args.timesteps_per_phase <= 0:
        parser.error("--rounds and --timesteps-per-phase must be positive")
    if args.v1_seeker.suffix != ".zip" or not args.v1_seeker.is_file():
        parser.error("--v1-seeker must point to an existing .zip model")
    return args


def load_v1_config(model_path: Path) -> tuple[int, int]:
    data = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("V1 model metadata must be an object")
    grid_size = data.get("grid_size")
    max_steps = data.get("max_steps")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("Invalid grid_size in V1 model metadata")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("Invalid max_steps in V1 model metadata")
    return grid_size, max_steps


def save_model(
    model: PPO,
    path: Path,
    *,
    role: str,
    grid_size: int,
    max_steps: int,
    seed: int,
    rounds: int,
    timesteps_per_phase: int,
    v1_seeker: Path,
    initial_timesteps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "role": role,
                "grid_size": grid_size,
                "max_steps": max_steps,
                "seed": seed,
                "rounds": rounds,
                "timesteps_per_phase": timesteps_per_phase,
                "actual_timesteps": model.num_timesteps,
                "actual_v2_timesteps": model.num_timesteps - initial_timesteps,
                "v1_seeker": str(v1_seeker),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    grid_size, max_steps = load_v1_config(args.v1_seeker)
    seeker_model = PPO.load(str(args.v1_seeker), device="cpu")
    v1_timesteps = seeker_model.num_timesteps
    seeker_model.set_random_seed(args.seed)

    hider_env = HideSeekEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        role="hider",
        opponent_policy=model_policy(seeker_model),
    )
    seeker_env = HideSeekEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        role="seeker",
    )
    hider_env.reset(seed=args.seed)
    seeker_env.reset(seed=args.seed + 1)
    hider_model = PPO(
        "MlpPolicy",
        hider_env,
        seed=args.seed,
        n_steps=512,
        batch_size=64,
        verbose=0,
    )
    seeker_model.set_env(seeker_env)

    try:
        for round_number in range(1, args.rounds + 1):
            print(f"Round {round_number}/{args.rounds}: training hider")
            hider_log = (
                args.output_dir / "results" / f"v2_hider_round_{round_number}.csv"
            )
            hider_model.learn(
                total_timesteps=args.timesteps_per_phase,
                callback=EpisodeLogger(hider_log),
                reset_num_timesteps=False,
            )

            print(f"Round {round_number}/{args.rounds}: training seeker")
            seeker_env.opponent_policy = model_policy(hider_model)
            seeker_log = (
                args.output_dir / "results" / f"v2_seeker_round_{round_number}.csv"
            )
            seeker_model.learn(
                total_timesteps=args.timesteps_per_phase,
                callback=EpisodeLogger(seeker_log),
                reset_num_timesteps=False,
            )

        save_model(
            hider_model,
            args.output_dir / "models/hider_v2.zip",
            role="hider",
            grid_size=grid_size,
            max_steps=max_steps,
            seed=args.seed,
            rounds=args.rounds,
            timesteps_per_phase=args.timesteps_per_phase,
            v1_seeker=args.v1_seeker,
            initial_timesteps=0,
        )
        save_model(
            seeker_model,
            args.output_dir / "models/seeker_v2.zip",
            role="seeker",
            grid_size=grid_size,
            max_steps=max_steps,
            seed=args.seed,
            rounds=args.rounds,
            timesteps_per_phase=args.timesteps_per_phase,
            v1_seeker=args.v1_seeker,
            initial_timesteps=v1_timesteps,
        )
    finally:
        hider_env.close()
        seeker_env.close()
    print(f"V2 models and logs: {args.output_dir}")


if __name__ == "__main__":
    main()
