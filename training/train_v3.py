"""Train independent V3 seeker/hider pairs on the room map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from env.hide_seek_v3_env import DEFAULT_MAP, HideSeekV3Env
from training.train import EpisodeLogger
from training.train_v2 import model_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V3 agents with walls and visibility")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--baseline-timesteps", type=int, default=100_000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timesteps-per-phase", type=int, default=50_000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if not args.training_seeds or len(args.training_seeds) != len(set(args.training_seeds)):
        parser.error("--training-seeds must be a nonempty list without repeats")
    if any(seed < 0 for seed in args.training_seeds):
        parser.error("--training-seeds must be nonnegative")
    if min(args.baseline_timesteps, args.rounds, args.timesteps_per_phase, args.max_steps) <= 0:
        parser.error("training durations, rounds, and --max-steps must be positive")
    return args


def save_model(
    model: PPO,
    path: Path,
    *,
    role: str,
    stage: str,
    seed: int,
    max_steps: int,
    baseline_timesteps: int,
    rounds: int,
    timesteps_per_phase: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": 3,
                "role": role,
                "stage": stage,
                "seed": seed,
                "map_rows": DEFAULT_MAP,
                "max_steps": max_steps,
                "baseline_timesteps": baseline_timesteps,
                "rounds": rounds,
                "timesteps_per_phase": timesteps_per_phase,
                "actual_timesteps": model.num_timesteps,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def train_seed(args: argparse.Namespace, seed: int) -> None:
    model_dir = args.output_dir / "models"
    result_dir = args.output_dir / "results"
    seeker_env = HideSeekV3Env(max_steps=args.max_steps, role="seeker")
    hider_env = HideSeekV3Env(max_steps=args.max_steps, role="hider")
    seeker_env.reset(seed=seed)
    hider_env.reset(seed=seed + 1000)
    seeker_model = PPO(
        "MlpPolicy",
        seeker_env,
        seed=seed,
        n_steps=512,
        batch_size=64,
        verbose=0,
        device="cpu",
    )
    try:
        print(f"Seed {seed}: seeker baseline against random hider", flush=True)
        seeker_model.learn(
            total_timesteps=args.baseline_timesteps,
            callback=EpisodeLogger(result_dir / f"v3_seed{seed}_seeker_baseline.csv"),
            reset_num_timesteps=False,
        )
        save_model(
            seeker_model,
            model_dir / f"seeker_v3_baseline_seed{seed}.zip",
            role="seeker",
            stage="baseline",
            seed=seed,
            max_steps=args.max_steps,
            baseline_timesteps=args.baseline_timesteps,
            rounds=args.rounds,
            timesteps_per_phase=args.timesteps_per_phase,
        )

        hider_env.opponent_policy = model_policy(seeker_model)
        hider_model = PPO(
            "MlpPolicy",
            hider_env,
            seed=seed,
            n_steps=512,
            batch_size=64,
            verbose=0,
            device="cpu",
        )
        for round_number in range(1, args.rounds + 1):
            print(f"Seed {seed}, round {round_number}: training hider", flush=True)
            hider_model.learn(
                total_timesteps=args.timesteps_per_phase,
                callback=EpisodeLogger(
                    result_dir / f"v3_seed{seed}_hider_round_{round_number}.csv"
                ),
                reset_num_timesteps=False,
            )
            seeker_env.opponent_policy = model_policy(hider_model)
            print(f"Seed {seed}, round {round_number}: training seeker", flush=True)
            seeker_model.learn(
                total_timesteps=args.timesteps_per_phase,
                callback=EpisodeLogger(
                    result_dir / f"v3_seed{seed}_seeker_round_{round_number}.csv"
                ),
                reset_num_timesteps=False,
            )

        save_model(
            hider_model,
            model_dir / f"hider_v3_seed{seed}.zip",
            role="hider",
            stage="final",
            seed=seed,
            max_steps=args.max_steps,
            baseline_timesteps=args.baseline_timesteps,
            rounds=args.rounds,
            timesteps_per_phase=args.timesteps_per_phase,
        )
        save_model(
            seeker_model,
            model_dir / f"seeker_v3_seed{seed}.zip",
            role="seeker",
            stage="final",
            seed=seed,
            max_steps=args.max_steps,
            baseline_timesteps=args.baseline_timesteps,
            rounds=args.rounds,
            timesteps_per_phase=args.timesteps_per_phase,
        )
    finally:
        seeker_env.close()
        hider_env.close()


def main() -> None:
    args = parse_args()
    for seed in args.training_seeds:
        train_seed(args, seed)
    print(f"V3 models and logs: {args.output_dir}")


if __name__ == "__main__":
    main()
