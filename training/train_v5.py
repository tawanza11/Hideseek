"""Compare equal-budget PPO training with and without legal action masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from env.hide_seek_v3_env import DEFAULT_MAP
from env.hide_seek_v4_env import HideSeekV4Env
from env.hide_seek_v5_env import HideSeekV5Env
from env.v4_maps import TEST_MAPS as V4_TEST_MAPS
from env.v4_maps import TRAIN_MAPS as V4_TRAIN_MAPS
from env.v4_maps import VALIDATION_MAPS as V4_VALIDATION_MAPS
from env.v5_maps import TRAIN_MAPS as V5_TRAIN_MAPS
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v4_models import load_v4_model
from training.train import EpisodeLogger
from training.train_v2 import model_policy


VARIANTS = ("control", "masked")
KNOWN_MAPS = {
    "default": DEFAULT_MAP,
    **V4_TRAIN_MAPS,
    **V4_VALIDATION_MAPS,
    **V4_TEST_MAPS,
}
# Previously opened V4 test maps are now training data. The V6 test split is new.
TRAINING_MAP_NAMES = (
    *("default" for _ in range(10)),
    *("F1" for _ in range(10)),
    *KNOWN_MAPS,
    *V5_TRAIN_MAPS,
)
TRAINING_MAPS = tuple(
    (KNOWN_MAPS | V5_TRAIN_MAPS)[name] for name in TRAINING_MAP_NAMES
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V5 control and masked seekers")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--novelty-bonus", type=float, default=0.01)
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v4-model-dir", type=Path, default=Path("models"))
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


def initialize_from_v4(model: PPO | MaskablePPO, source: PPO) -> None:
    """Give both variants the same initial actor and critic, with fresh optimizers."""
    original = source.policy.state_dict()
    target = model.policy.state_dict()
    if original.keys() != target.keys() or any(
        original[name].shape != target[name].shape for name in original
    ):
        raise ValueError("V4 and V5 policy structures differ")
    with torch.no_grad():
        model.policy.load_state_dict(original)


def train_variant(
    *,
    seed: int,
    variant: str,
    timesteps: int,
    novelty_bonus: float,
    v3_model_dir: Path,
    v4_model_dir: Path,
    output_dir: Path,
) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown V5 variant: {variant}")
    source_path = v4_model_dir / f"seeker_v4_nomemory_seed{seed}.zip"
    hider_path = v3_model_dir / f"hider_v3_seed{seed}.zip"
    hider, max_steps = load_model(hider_path, role="hider", stage="final", seed=seed)
    source = load_v4_model(source_path, variant="nomemory", seed=seed, max_steps=max_steps)
    common = {
        "map_pool": TRAINING_MAPS,
        "novelty_bonus": novelty_bonus,
        "max_steps": max_steps,
        "opponent_policy": model_policy(hider),
    }
    env = (
        HideSeekV5Env(**common)
        if variant == "masked"
        else HideSeekV4Env(**common, visit_memory=False)
    )
    env.reset(seed=seed + 5000)
    model_class = MaskablePPO if variant == "masked" else PPO
    model = model_class(
        "MlpPolicy", env, seed=seed + 5000,
        n_steps=source.n_steps, batch_size=source.batch_size,
        verbose=0, device="cpu",
    )
    initialize_from_v4(model, source)
    log_path = output_dir / "results" / f"v5_seed{seed}_{variant}_episodes.csv"
    try:
        print(f"Seed {seed}: training V5 {variant} seeker", flush=True)
        model.learn(total_timesteps=timesteps, callback=EpisodeLogger(log_path))
        model_path = output_dir / "models" / f"seeker_v5_{variant}_seed{seed}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        model_path.with_suffix(".json").write_text(
            json.dumps({
                "version": 5,
                "role": "seeker",
                "variant": variant,
                "seed": seed,
                "action_mask_during_training": variant == "masked",
                "novelty_bonus": novelty_bonus,
                "training_map_names": TRAINING_MAP_NAMES,
                "distinct_training_maps": len(set(TRAINING_MAP_NAMES)),
                "max_steps": max_steps,
                "source_model": str(source_path),
                "hider_model": str(hider_path),
                "source_timesteps": source.num_timesteps,
                "requested_timesteps": timesteps,
                "actual_timesteps": model.num_timesteps,
                "optimizer": "fresh_for_both_variants",
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    for seed in args.training_seeds:
        for variant in VARIANTS:
            train_variant(
                seed=seed, variant=variant, timesteps=args.timesteps,
                novelty_bonus=args.novelty_bonus,
                v3_model_dir=args.v3_model_dir,
                v4_model_dir=args.v4_model_dir,
                output_dir=args.output_dir,
            )
    print(f"V5 models and logs: {args.output_dir}")


if __name__ == "__main__":
    main()
