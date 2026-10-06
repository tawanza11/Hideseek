"""Watch or record one saved V4/V5/V6 Seeker game against a V3 Hider."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env.hide_seek_v3_env import DEFAULT_MAP, HideSeekV3Env
from env.v5_maps import FINAL_TEST_MAPS, VALIDATION_MAPS
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v31 import masked_action
from evaluation.evaluate_v4_models import load_v4_model
from evaluation.evaluate_v5 import load_v5_model
from evaluation.evaluate_v6 import load_v6_model
from training.train_v2 import model_policy


MODEL_NAMES = ("v4", "v5", "v6-flat", "v6-curriculum")
MAPS = {"default": DEFAULT_MAP, **VALIDATION_MAPS, **FINAL_TEST_MAPS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch one saved Hide-and-Seek game")
    parser.add_argument("--model", choices=MODEL_NAMES, default="v6-curriculum")
    parser.add_argument("--map", choices=tuple(MAPS), default="V5F001")
    parser.add_argument("--training-seed", type=int, choices=(41, 42, 43), default=41)
    parser.add_argument("--episode-seed", type=int)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--gif", type=Path, help="Save an animation instead of opening a window")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    if args.episode_seed is not None and args.episode_seed < 0:
        parser.error("--episode-seed must be nonnegative")
    if not 1 <= args.fps <= 30:
        parser.error("--fps must be between 1 and 30")
    return args


def default_episode_seed(map_name: str) -> int:
    if map_name == "default":
        return 500_000
    if map_name in VALIDATION_MAPS:
        return 400_000 + list(VALIDATION_MAPS).index(map_name) * 10_000
    return 800_000 + list(FINAL_TEST_MAPS).index(map_name) * 10_000


def load_seeker(model_dir: Path, name: str, seed: int, max_steps: int) -> PPO:
    if name == "v4":
        return load_v4_model(
            model_dir / f"seeker_v4_nomemory_seed{seed}.zip",
            variant="nomemory", seed=seed, max_steps=max_steps,
        )
    if name == "v5":
        model = load_v5_model(
            model_dir / f"seeker_v5_control_seed{seed}.zip",
            variant="control", seed=seed, max_steps=max_steps,
        )
        if not isinstance(model, PPO):
            raise TypeError("V5 control model is not PPO")
        return model
    variant = name.removeprefix("v6-")
    return load_v6_model(
        model_dir / f"seeker_v6_{variant}_seed{seed}.zip",
        variant=variant, seed=seed, max_steps=max_steps,
    )


def save_gif(
    frames: list[np.ndarray], path: Path, *, fps: int, model_name: str, map_name: str
) -> None:
    from PIL import Image, ImageDraw

    if not frames:
        raise ValueError("Cannot save a game with no frames")
    images: list[Image.Image] = []
    for step, frame in enumerate(frames):
        image = Image.new("RGB", (frame.shape[1], frame.shape[0] + 44), "white")
        image.paste(Image.fromarray(frame), (0, 0))
        draw = ImageDraw.Draw(image)
        draw.text(
            (10, frame.shape[0] + 7),
            f"{model_name} | {map_name} | step {step} | blue: seeker, green: hider",
            fill="black",
        )
        images.append(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=round(1000 / fps), loop=0, optimize=True,
    )


def main() -> None:
    args = parse_args()
    episode_seed = (
        default_episode_seed(args.map)
        if args.episode_seed is None else args.episode_seed
    )
    hider, max_steps = load_model(
        args.model_dir / f"hider_v3_seed{args.training_seed}.zip",
        role="hider", stage="final", seed=args.training_seed,
    )
    seeker = load_seeker(args.model_dir, args.model, args.training_seed, max_steps)
    env = HideSeekV3Env(
        map_rows=MAPS[args.map], role="seeker", max_steps=max_steps,
        opponent_policy=model_policy(hider),
        render_mode="rgb_array" if args.gif else "human",
    )
    env.metadata = {**env.metadata, "render_fps": args.fps}
    observation, _ = env.reset(seed=episode_seed)
    frames: list[np.ndarray] = []

    def show_frame() -> None:
        frame = env.render()
        if frame is not None:
            frames.append(frame.copy())

    try:
        for step in range(1, max_steps + 1):
            show_frame()
            action = masked_action(seeker, observation)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                show_frame()
                outcome = "Seeker captured Hider" if info["captured"] else "Hider survived"
                print(f"{outcome} after {step} steps | {args.model} | {args.map} | seed {episode_seed}")
                break
    except KeyboardInterrupt:
        print("Viewer closed")
        return
    finally:
        env.close()
    if args.gif:
        save_gif(frames, args.gif, fps=args.fps, model_name=args.model, map_name=args.map)
        print(f"Animation: {args.gif}")


if __name__ == "__main__":
    main()
