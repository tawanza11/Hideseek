"""Compare V4 seekers with and without a visited-cell observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from env.hide_seek_v3_env import DEFAULT_MAP
from env.hide_seek_v4_env import HideSeekV4Env
from env.v4_maps import TRAIN_MAPS
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v31 import load_v31_model
from training.train import EpisodeLogger
from training.train_v2 import model_policy


VARIANTS = ("nomemory", "memory")
TRAINING_MAPS = (DEFAULT_MAP, *TRAIN_MAPS.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V4 seekers on three room maps")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--novelty-bonus", type=float, default=0.01)
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v31-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if not args.training_seeds or any(seed < 0 for seed in args.training_seeds):
        parser.error("--training-seeds must be nonempty and nonnegative")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("--training-seeds must be unique")
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if not 0 <= args.novelty_bonus < 1:
        parser.error("--novelty-bonus must be in [0, 1)")
    return args


def initialize_from_v31(model: PPO, source: PPO) -> None:
    """Copy policy weights, leaving new memory input weights at zero."""
    source_state = source.policy.state_dict()
    target_state = model.policy.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("V3.1 and V4 policy structures differ")
    first_layers = {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    }
    with torch.no_grad():
        for name, target in target_state.items():
            original = source_state[name]
            if target.shape == original.shape:
                target.copy_(original)
            elif (
                name in first_layers
                and target.ndim == 2
                and original.ndim == 2
                and target.shape[0] == original.shape[0]
                and target.shape[1] == original.shape[1] + 100
            ):
                target.zero_()
                target[:, : original.shape[1]].copy_(original)
            else:
                raise ValueError(f"Cannot initialize V4 policy tensor {name}")
    model.policy.load_state_dict(target_state)


def train_variant(
    *,
    seed: int,
    variant: str,
    timesteps: int,
    novelty_bonus: float,
    v3_model_dir: Path,
    v31_model_dir: Path,
    output_dir: Path,
) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown V4 variant: {variant}")
    source_path = v31_model_dir / f"seeker_v31_control_seed{seed}.zip"
    hider_path = v3_model_dir / f"hider_v3_seed{seed}.zip"
    hider, max_steps = load_model(hider_path, role="hider", stage="final", seed=seed)
    source = load_v31_model(source_path, seed=seed, variant="control", max_steps=max_steps)
    env = HideSeekV4Env(
        map_pool=TRAINING_MAPS,
        visit_memory=variant == "memory",
        novelty_bonus=novelty_bonus,
        max_steps=max_steps,
        opponent_policy=model_policy(hider),
    )
    env.reset(seed=seed + 3000)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed + 3000,
        n_steps=source.n_steps,
        batch_size=source.batch_size,
        verbose=0,
        device="cpu",
    )
    initialize_from_v31(model, source)
    log_path = output_dir / "results" / f"v4_seed{seed}_{variant}_episodes.csv"
    try:
        print(f"Seed {seed}: training V4 {variant} seeker", flush=True)
        model.learn(total_timesteps=timesteps, callback=EpisodeLogger(log_path))
        model_path = output_dir / "models" / f"seeker_v4_{variant}_seed{seed}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        model_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "version": 4,
                    "role": "seeker",
                    "variant": variant,
                    "seed": seed,
                    "visit_memory": variant == "memory",
                    "novelty_bonus": novelty_bonus,
                    "training_maps": ["default", *TRAIN_MAPS.keys()],
                    "map_rows": TRAINING_MAPS,
                    "max_steps": max_steps,
                    "source_model": str(source_path),
                    "hider_model": str(hider_path),
                    "source_timesteps": source.num_timesteps,
                    "requested_timesteps": timesteps,
                    "actual_timesteps": model.num_timesteps,
                    "optimizer": "fresh_for_both_variants",
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
        for variant in VARIANTS:
            train_variant(
                seed=seed,
                variant=variant,
                timesteps=args.timesteps,
                novelty_bonus=args.novelty_bonus,
                v3_model_dir=args.v3_model_dir,
                v31_model_dir=args.v31_model_dir,
                output_dir=args.output_dir,
            )
    print(f"V4 models and logs: {args.output_dir}")


if __name__ == "__main__":
    main()
