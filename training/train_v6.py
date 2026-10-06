"""Compare equal-budget flat and curriculum map training against a Hider pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from env.v5_maps import TRAIN_MAPS as V5_TRAIN_MAPS
from env.hide_seek_v6_env import HideSeekV6Env
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v4_models import load_v4_model
from training.train import EpisodeLogger
from training.train_v2 import model_policy
from training.train_v5 import KNOWN_MAPS, TRAINING_MAP_NAMES, TRAINING_MAPS, initialize_from_v4


VARIANTS = ("flat", "curriculum")
HIDER_SEEDS = (41, 42, 43)
KNOWN_POOL = tuple(KNOWN_MAPS[name] for name in TRAINING_MAP_NAMES if name in KNOWN_MAPS)
PROCEDURAL_POOL = tuple(V5_TRAIN_MAPS.values())
STAGE_COUNTS = (20, 50, 100)
assert KNOWN_POOL + PROCEDURAL_POOL == TRAINING_MAPS


def map_pool(variant: str, stage: int) -> tuple[tuple[str, ...], ...]:
    if variant not in VARIANTS or not 0 <= stage < len(STAGE_COUNTS):
        raise ValueError("Invalid V6 variant or stage")
    count = STAGE_COUNTS[stage] if variant == "curriculum" else len(PROCEDURAL_POOL)
    return KNOWN_POOL + PROCEDURAL_POOL[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V6 flat and curriculum seekers")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=list(HIDER_SEEDS))
    parser.add_argument("--phase-timesteps", type=int, nargs=3, default=[50_000, 50_000, 100_000])
    parser.add_argument("--novelty-bonus", type=float, default=0.01)
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v4-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if not args.training_seeds or any(seed < 0 for seed in args.training_seeds):
        parser.error("--training-seeds must be nonempty and nonnegative")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("--training-seeds must be unique")
    if any(steps <= 0 for steps in args.phase_timesteps):
        parser.error("--phase-timesteps values must be positive")
    if not 0 <= args.novelty_bonus < 1:
        parser.error("--novelty-bonus must be in [0, 1)")
    return args


def train_variant(args: argparse.Namespace, seed: int, variant: str) -> None:
    source_path = args.v4_model_dir / f"seeker_v4_nomemory_seed{seed}.zip"
    hider_paths = tuple(
        args.v3_model_dir / f"hider_v3_seed{hider_seed}.zip"
        for hider_seed in HIDER_SEEDS
    )
    hiders = [
        load_model(path, role="hider", stage="final", seed=hider_seed)
        for path, hider_seed in zip(hider_paths, HIDER_SEEDS, strict=True)
    ]
    max_steps = hiders[0][1]
    if any(steps != max_steps for _, steps in hiders):
        raise ValueError("Hider max_steps differ")
    source = load_v4_model(source_path, variant="nomemory", seed=seed, max_steps=max_steps)
    env = HideSeekV6Env(
        map_pool=map_pool(variant, 0),
        novelty_bonus=args.novelty_bonus,
        max_steps=max_steps,
        opponent_policies=(None, *(model_policy(model) for model, _ in hiders)),
    )
    env.reset(seed=seed + 6000)
    model = PPO(
        "MlpPolicy", env, seed=seed + 6000,
        n_steps=source.n_steps, batch_size=source.batch_size,
        verbose=0, device="cpu",
    )
    initialize_from_v4(model, source)
    phase_actual_steps: list[int] = []
    try:
        for stage, requested_steps in enumerate(args.phase_timesteps):
            env.map_pool = map_pool(variant, stage)
            model.set_env(env, force_reset=True)
            before = model.num_timesteps
            log_path = args.output_dir / "results" / f"v6_seed{seed}_{variant}_stage{stage + 1}.csv"
            print(f"Seed {seed}: V6 {variant} stage {stage + 1}/3", flush=True)
            model.learn(
                total_timesteps=requested_steps,
                callback=EpisodeLogger(log_path),
                reset_num_timesteps=False,
            )
            phase_actual_steps.append(model.num_timesteps - before)
        model_path = args.output_dir / "models" / f"seeker_v6_{variant}_seed{seed}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        model_path.with_suffix(".json").write_text(
            json.dumps({
                "version": 6,
                "role": "seeker",
                "variant": variant,
                "seed": seed,
                "source_model": str(source_path),
                "source_timesteps": source.num_timesteps,
                "hider_models": [str(path) for path in hider_paths],
                "opponent_pool": ["random", *[f"hider_v3_seed{s}" for s in HIDER_SEEDS]],
                "stage_map_counts": [len(set(map_pool(variant, stage))) for stage in range(3)],
                "phase_requested_timesteps": args.phase_timesteps,
                "phase_actual_timesteps": phase_actual_steps,
                "actual_timesteps": model.num_timesteps,
                "novelty_bonus": args.novelty_bonus,
                "max_steps": max_steps,
                "optimizer": "fresh_from_V4_weights",
                "action_mask_during_training": False,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    for seed in args.training_seeds:
        for variant in VARIANTS:
            train_variant(args, seed, variant)
    print(f"V6 models and logs: {args.output_dir}")


if __name__ == "__main__":
    main()
