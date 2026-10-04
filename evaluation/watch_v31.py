"""Show one V3.1 masked Seeker game against the saved V3 Hider."""

from __future__ import annotations

import argparse
from pathlib import Path

from env.hide_seek_v3_env import HideSeekV3Env
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v31 import load_v31_model, masked_action
from training.train_v2 import model_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch one V3.1 Hide-and-Seek game")
    parser.add_argument("--training-seed", type=int, default=41)
    parser.add_argument("--episode-seed", type=int, default=70069)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v31-model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    if args.training_seed < 0 or args.episode_seed < 0 or args.fps <= 0:
        parser.error("seeds must be nonnegative and --fps must be positive")

    hider, max_steps = load_model(
        args.v3_model_dir / f"hider_v3_seed{args.training_seed}.zip",
        role="hider",
        stage="final",
        seed=args.training_seed,
    )
    seeker = load_v31_model(
        args.v31_model_dir / f"seeker_v31_control_seed{args.training_seed}.zip",
        seed=args.training_seed,
        variant="control",
        max_steps=max_steps,
    )
    env = HideSeekV3Env(
        role="seeker",
        max_steps=max_steps,
        opponent_policy=model_policy(hider),
        render_mode="human",
    )
    env.metadata = {**env.metadata, "render_fps": args.fps}
    observation, _ = env.reset(seed=args.episode_seed)
    try:
        for step in range(1, max_steps + 1):
            env.render()
            action = masked_action(seeker, observation)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.render()
                print(f"{'Captured' if info['captured'] else 'Hider survived'} in {step} steps")
                return
    finally:
        env.close()


if __name__ == "__main__":
    main()
