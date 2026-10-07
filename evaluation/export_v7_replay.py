"""Export a real V7 match with dynamic objects for the 3D spectator view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from env.hide_seek_v7_env import HideSeekV7Env
from env.v7_maps import TRAIN_LAYOUTS, VALIDATION_LAYOUTS, layout_hash
from evaluation.evaluate_v7_seeker import learned_action


MAPS = {**TRAIN_LAYOUTS, **VALIDATION_LAYOUTS}
ACTION_NAMES = ("up", "down", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a V7 object match as JSON")
    parser.add_argument("--map", choices=tuple(MAPS), default="V7E001")
    parser.add_argument("--training-seed", type=int, choices=(41, 42, 43), default=41)
    parser.add_argument("--episode-seed", type=int, default=1_700_000)
    parser.add_argument("--seeker-variant", choices=("route", "raw"), default="route")
    parser.add_argument("--seeker-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--hider-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episode_seed < 0:
        parser.error("--episode-seed must be nonnegative")
    return args


def load_v7_model(path: Path, role: str, seed: int, expected_size: int) -> PPO:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if ((metadata.get("version"), metadata.get("role"), metadata.get("seed")) != (7, role, seed)
            or metadata.get("training_layout_hash") != layout_hash(TRAIN_LAYOUTS)):
        raise ValueError(f"V7 model metadata mismatch: {path}")
    model = PPO.load(str(path), device="cpu")
    if model.observation_space.shape != (expected_size,):
        raise ValueError(f"V7 observation mismatch: {path}")
    return model


def export_replay(
    map_name: str, seed: int, episode_seed: int,
    seeker_variant: str, seeker_model_dir: Path, hider_model_dir: Path,
) -> dict[str, object]:
    seeker_prefix = "seeker_v7_route" if seeker_variant == "route" else "seeker_v7"
    seeker = load_v7_model(
        seeker_model_dir / f"{seeker_prefix}_seed{seed}.zip", "seeker", seed,
        516 if seeker_variant == "route" else 508,
    )
    hider = load_v7_model(hider_model_dir / f"hider_v7_seed{seed}.zip", "hider", seed, 508)
    env = HideSeekV7Env(
        layouts=(MAPS[map_name],), role="seeker",
        opponent_policy=lambda observation: learned_action(hider, observation),
    )
    frames: list[dict[str, object]] = []

    def frame(step: int, action: int | None = None, event: str | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "step": step,
            "seeker": list(env.seeker_pos),
            "hider": list(env.hider_pos),
            "visible": env.is_visible("seeker"),
            "blocks": [list(cell) for cell in sorted(env.blocks)],
            "ramps": [list(env.ramp)],
            "low_walls": [list(env.low_wall)],
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
        for step in range(1, env.max_steps + 1):
            action = learned_action(seeker, observation)
            observation, _, terminated, truncated, info = env.step(action)
            outcome = "seeker_captured" if info["captured"] else "hider_survived" if truncated else ""
            if outcome:
                event = outcome
            elif env.last_events:
                actor, kind = env.last_events[0]
                event = f"{actor}_{kind}"
            else:
                event = None
            frames.append(frame(step, action, event))
            if terminated or truncated:
                break
        if not outcome:
            raise RuntimeError("Episode ended without an outcome")
        return {
            "schema_version": 1,
            "model": f"{seeker_prefix}_seed{seed}",
            "hider_model": f"hider_v7_seed{seed}",
            "map": map_name,
            "training_seed": seed,
            "episode_seed": episode_seed,
            "max_steps": env.max_steps,
            "outcome": outcome,
            "map_rows": list(env.map_rows),
            "frames": frames,
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    replay = export_replay(
        args.map, args.training_seed, args.episode_seed,
        args.seeker_variant, args.seeker_model_dir, args.hider_model_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(replay['frames'])} frames, {replay['outcome']}: {args.output}")


if __name__ == "__main__":
    main()
