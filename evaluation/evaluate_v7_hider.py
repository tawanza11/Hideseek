"""Paired V7 Hider comparison against random and observable-search Seekers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env.hide_seek_v7_env import HideSeekV7Env, ObjectLayout
from env.v7_maps import TRAIN_LAYOUTS, VALIDATION_LAYOUTS, layout_hash
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v7_seeker import learned_action
from training.v7_teacher import legal_actions, search_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V7 Hider vs fixed Seekers")
    parser.add_argument("--training-seed", type=int, choices=(41, 42, 43), default=41)
    parser.add_argument("--episodes-per-map", type=int, default=100)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("results/v7_hider_validation.csv"))
    args = parser.parse_args()
    if args.episodes_per_map < 1:
        parser.error("--episodes-per-map must be positive")
    return args


def run_game(
    layout: ObjectLayout, seed: int, seeker_name: str, hider_name: str,
    v3_hider: PPO, v7_hider: PPO,
    *,
    enable_blocks: bool = True,
    enable_ramp: bool = True,
    spawn: dict[str, tuple[int, int]] | None = None,
) -> dict[str, int]:
    env = HideSeekV7Env(
        layouts=(layout,), role="hider",
        opponent_policy=search_action if seeker_name == "teacher" else None,
        enable_blocks=enable_blocks, enable_ramp=enable_ramp,
    )
    observation, _ = env.reset(seed=seed, options=spawn)
    hider_rng = np.random.default_rng(seed + 29)
    pushes = crosses = 0
    survived = False
    try:
        for step in range(1, env.max_steps + 1):
            if hider_name == "random":
                allowed = legal_actions(observation)
                action = int(hider_rng.choice(allowed)) if allowed else 0
            elif hider_name == "v3":
                action, _ = v3_hider.predict(observation[:105], deterministic=True)
                action = int(action)
            elif hider_name == "v7":
                action = learned_action(v7_hider, observation)
            else:
                raise ValueError(f"invalid Hider: {hider_name}")
            observation, _, terminated, truncated, _ = env.step(action)
            pushes += int(("hider", "block_pushed") in env.last_events)
            crosses += int(("hider", "climbed") in env.last_events)
            if terminated or truncated:
                survived = truncated
                break
        return {"survived": int(survived), "steps": step, "pushes": pushes, "crosses": crosses}
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    model_path = args.model_dir / f"hider_v7_seed{args.training_seed}.zip"
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    if (metadata.get("version"), metadata.get("role"), metadata.get("seed")) != (
        7, "hider", args.training_seed
    ) or metadata.get("training_layout_hash") != layout_hash(TRAIN_LAYOUTS):
        raise ValueError("V7 Hider metadata mismatch")
    v7_hider = PPO.load(str(model_path), device="cpu")
    if v7_hider.observation_space.shape != (508,):
        raise ValueError("V7 Hider observation mismatch")
    v3_hider, _ = load_model(
        args.v3_model_dir / f"hider_v3_seed{args.training_seed}.zip",
        role="hider", stage="final", seed=args.training_seed,
    )
    rows: list[dict[str, str | int]] = []
    for map_index, (name, layout) in enumerate(VALIDATION_LAYOUTS.items()):
        counts = {(seeker, hider): 0 for seeker in ("random", "teacher")
                  for hider in ("random", "v3", "v7")}
        for episode in range(args.episodes_per_map):
            seed = 1_600_000 + map_index * 10_000 + episode
            for seeker, hider in counts:
                result = run_game(layout, seed, seeker, hider, v3_hider, v7_hider)
                rows.append({
                    "map": name, "training_seed": args.training_seed,
                    "eval_seed": seed, "seeker": seeker, "hider": hider, **result,
                })
                counts[(seeker, hider)] += result["survived"]
        print(name, counts, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "map", "training_seed", "eval_seed", "seeker", "hider", "survived", "steps", "pushes", "crosses"
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
