"""Paired V3.1 experiment: continued seeker training with/without wall penalty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from env.hide_seek_v3_env import DEFAULT_MAP, HideSeekV3Env
from env.hide_seek_v31_env import SeekerWallPenaltyEnv
from evaluation.evaluate_v3 import load_model
from training.train import EpisodeLogger
from training.train_v2 import model_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare V3.1 seeker wall-penalty training")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--wall-penalty", type=float, default=0.02)
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if any(seed < 0 for seed in args.training_seeds) or len(set(args.training_seeds)) != len(
        args.training_seeds
    ):
        parser.error("--training-seeds must be nonnegative and unique")
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if not 0 < args.wall_penalty < 1:
        parser.error("--wall-penalty must be between 0 and 1")
    return args


def train_variant(
    *,
    seed: int,
    variant: str,
    timesteps: int,
    wall_penalty: float,
    v3_model_dir: Path,
    output_dir: Path,
) -> None:
    source_path = v3_model_dir / f"seeker_v3_seed{seed}.zip"
    hider_path = v3_model_dir / f"hider_v3_seed{seed}.zip"
    source, max_steps = load_model(source_path, role="seeker", stage="final", seed=seed)
    hider, hider_max_steps = load_model(hider_path, role="hider", stage="final", seed=seed)
    if max_steps != hider_max_steps:
        raise ValueError(f"V3 model time limits differ for seed {seed}")
    source_steps = source.num_timesteps
    opponent = model_policy(hider)
    if variant == "penalty":
        env = SeekerWallPenaltyEnv(
            max_steps=max_steps,
            opponent_policy=opponent,
            wall_penalty=wall_penalty,
        )
    elif variant == "control":
        env = HideSeekV3Env(
            max_steps=max_steps,
            role="seeker",
            opponent_policy=opponent,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    model = PPO.load(str(source_path), env=env, device="cpu")
    model.set_random_seed(seed + 2000)
    env.reset(seed=seed + 2000)
    log_path = output_dir / "results" / f"v31_seed{seed}_{variant}_episodes.csv"
    try:
        print(f"Seed {seed}: training {variant} seeker", flush=True)
        model.learn(
            total_timesteps=timesteps,
            callback=EpisodeLogger(log_path),
            reset_num_timesteps=False,
        )
        model_path = output_dir / "models" / f"seeker_v31_{variant}_seed{seed}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        model_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "version": "3.1",
                    "role": "seeker",
                    "variant": variant,
                    "seed": seed,
                    "wall_penalty": wall_penalty if variant == "penalty" else 0.0,
                    "map_rows": DEFAULT_MAP,
                    "max_steps": max_steps,
                    "source_model": str(source_path),
                    "hider_model": str(hider_path),
                    "source_timesteps": source_steps,
                    "additional_timesteps_requested": timesteps,
                    "actual_additional_timesteps": model.num_timesteps - source_steps,
                    "total_timesteps": model.num_timesteps,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    for seed in args.training_seeds:
        for variant in ("control", "penalty"):
            train_variant(
                seed=seed,
                variant=variant,
                timesteps=args.timesteps,
                wall_penalty=args.wall_penalty,
                v3_model_dir=args.v3_model_dir,
                output_dir=args.output_dir,
            )
    print(f"V3.1 models and logs: {args.output_dir}")


if __name__ == "__main__":
    main()
