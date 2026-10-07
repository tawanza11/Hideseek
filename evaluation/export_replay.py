"""Export one real saved-model match for the browser 3D viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from env.hide_seek_v3_env import HideSeekV3Env
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v31 import masked_action
from evaluation.watch_v6 import MAPS, MODEL_NAMES, default_episode_seed, load_seeker
from training.train_v2 import model_policy


ACTION_NAMES = ("up", "down", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a saved Hide-and-Seek match as JSON")
    parser.add_argument("--model", choices=MODEL_NAMES, default="v6-curriculum")
    parser.add_argument("--map", choices=tuple(MAPS), default="V5F001")
    parser.add_argument("--training-seed", type=int, choices=(41, 42, 43), default=41)
    parser.add_argument("--episode-seed", type=int)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episode_seed is not None and args.episode_seed < 0:
        parser.error("--episode-seed must be nonnegative")
    return args


def export_replay(
    *, model_name: str, map_name: str, training_seed: int,
    episode_seed: int, model_dir: Path,
) -> dict[str, object]:
    hider, max_steps = load_model(
        model_dir / f"hider_v3_seed{training_seed}.zip",
        role="hider", stage="final", seed=training_seed,
    )
    seeker = load_seeker(model_dir, model_name, training_seed, max_steps)
    env = HideSeekV3Env(
        map_rows=MAPS[map_name], role="seeker", max_steps=max_steps,
        opponent_policy=model_policy(hider),
    )
    frames: list[dict[str, object]] = []

    def frame(step: int, action: int | None = None, event: str | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "step": step,
            "seeker": list(env.seeker_pos),
            "hider": list(env.hider_pos),
            "visible": env.is_visible("seeker"),
            "blocks": [],
            "ramps": [],
            "low_walls": [],
        }
        if action is not None:
            result["action"] = ACTION_NAMES[action]
        if event is not None:
            result["event"] = event
        return result

    try:
        observation, _ = env.reset(seed=episode_seed)
        frames.append(frame(0, event="start"))
        outcome = ""
        for step in range(1, max_steps + 1):
            action = masked_action(seeker, observation)
            observation, _, terminated, truncated, info = env.step(action)
            outcome = "seeker_captured" if info["captured"] else "hider_survived" if truncated else ""
            frames.append(frame(step, action=action, event=outcome if outcome else None))
            if terminated or truncated:
                break
        if not outcome:
            raise RuntimeError("Episode ended without an outcome")
        return {
            "schema_version": 1,
            "model": model_name,
            "hider_model": "v3",
            "map": map_name,
            "training_seed": training_seed,
            "episode_seed": episode_seed,
            "max_steps": max_steps,
            "outcome": outcome,
            "map_rows": list(env.map_rows),
            "frames": frames,
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    seed = default_episode_seed(args.map) if args.episode_seed is None else args.episode_seed
    replay = export_replay(
        model_name=args.model, map_name=args.map, training_seed=args.training_seed,
        episode_seed=seed, model_dir=args.model_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(replay['frames'])} frames, {replay['outcome']}: {args.output}")


if __name__ == "__main__":
    main()
