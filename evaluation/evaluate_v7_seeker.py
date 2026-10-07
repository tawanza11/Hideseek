"""Paired V7 Seeker comparison on new object layouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from env.hide_seek_v7_env import HideSeekV7Env, ObjectLayout
from env.v7_maps import TRAIN_LAYOUTS, VALIDATION_LAYOUTS, layout_hash
from evaluation.evaluate_v3 import load_model
from training.train_v2 import model_policy
from training.v7_observation import with_route_hints
from training.v7_teacher import legal_actions, search_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V7 seeker vs frozen V3 hider")
    parser.add_argument("--training-seed", type=int, choices=(41, 42, 43), default=41)
    parser.add_argument("--variant", choices=("raw", "route"), default="raw")
    parser.add_argument("--hider-variant", choices=("v3", "v7"), default="v3")
    parser.add_argument("--episodes-per-map", type=int, default=100)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v7-hider-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("results/v7_seeker_validation.csv"))
    args = parser.parse_args()
    if args.episodes_per_map < 1:
        parser.error("--episodes-per-map must be positive")
    return args


def learned_action(model: PPO, observation: np.ndarray) -> int:
    allowed = legal_actions(observation)
    if not allowed:
        return 0
    model_observation = (
        with_route_hints(observation)
        if model.observation_space.shape == (516,)
        else observation
    )
    with torch.no_grad():
        tensor = torch.as_tensor(model_observation[None, :], device=model.device)
        probabilities = model.policy.get_distribution(tensor).distribution.probs[0].cpu().numpy()
    return max(allowed, key=lambda action: float(probabilities[action]))


def run_game(
    layout: ObjectLayout,
    seed: int,
    hider_policy,
    matchup: str,
    model: PPO | None,
    *,
    enable_blocks: bool = True,
    enable_ramp: bool = True,
    spawn: dict[str, tuple[int, int]] | None = None,
) -> dict[str, int | str]:
    env = HideSeekV7Env(
        layouts=(layout,), role="seeker",
        opponent_policy=hider_policy,
        enable_blocks=enable_blocks, enable_ramp=enable_ramp,
    )
    observation, _ = env.reset(seed=seed, options=spawn)
    env.action_space.seed(seed + 17)
    pushes = crosses = 0
    outcome = False
    try:
        for step in range(1, env.max_steps + 1):
            if matchup == "random":
                allowed = legal_actions(observation)
                action = int(env.np_random.choice(allowed)) if allowed else 0
            elif matchup == "teacher":
                action = search_action(observation)
            elif matchup == "learned" and model is not None:
                action = learned_action(model, observation)
            else:
                raise ValueError(f"invalid matchup: {matchup}")
            observation, _, terminated, truncated, info = env.step(action)
            pushes += int(("seeker", "block_pushed") in env.last_events)
            crosses += int(("seeker", "climbed") in env.last_events)
            if terminated or truncated:
                outcome = bool(info["captured"])
                break
        return {"captured": int(outcome), "steps": step, "pushes": pushes, "crosses": crosses}
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    prefix = "seeker_v7_route" if args.variant == "route" else "seeker_v7"
    model_path = args.model_dir / f"{prefix}_seed{args.training_seed}.zip"
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    if (metadata.get("version"), metadata.get("role"), metadata.get("seed"), metadata.get("route_hints", False)) != (
        7, "seeker", args.training_seed, args.variant == "route"
    ) or metadata.get("training_layout_hash") != layout_hash(TRAIN_LAYOUTS):
        raise ValueError("V7 model metadata mismatch")
    model = PPO.load(str(model_path), device="cpu")
    expected_size = 516 if args.variant == "route" else 508
    if model.observation_space.shape != (expected_size,):
        raise ValueError("V7 model observation size mismatch")
    if args.hider_variant == "v3":
        hider, _ = load_model(
            args.v3_model_dir / f"hider_v3_seed{args.training_seed}.zip",
            role="hider", stage="final", seed=args.training_seed,
        )
        old_policy = model_policy(hider)
        hider_policy = lambda observation: old_policy(observation[:105])
    else:
        hider_path = args.v7_hider_model_dir / f"hider_v7_seed{args.training_seed}.zip"
        hider_metadata = json.loads(hider_path.with_suffix(".json").read_text(encoding="utf-8"))
        if (hider_metadata.get("version"), hider_metadata.get("role"), hider_metadata.get("seed")) != (
            7, "hider", args.training_seed
        ) or hider_metadata.get("training_layout_hash") != layout_hash(TRAIN_LAYOUTS):
            raise ValueError("V7 Hider metadata mismatch")
        hider = PPO.load(str(hider_path), device="cpu")
        if hider.observation_space.shape != (508,):
            raise ValueError("V7 Hider observation mismatch")
        hider_policy = lambda observation: learned_action(hider, observation)
    rows: list[dict[str, int | str]] = []
    for map_index, (name, layout) in enumerate(VALIDATION_LAYOUTS.items()):
        counts = {matchup: 0 for matchup in ("random", "teacher", "learned")}
        for episode in range(args.episodes_per_map):
            seed = 1_400_000 + map_index * 10_000 + episode
            for matchup in counts:
                result = run_game(layout, seed, hider_policy, matchup, model)
                rows.append({
                    "map": name, "training_seed": args.training_seed,
                    "eval_seed": seed, "matchup": matchup, **result,
                })
                counts[matchup] += int(result["captured"])
        print(name, counts, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "map", "training_seed", "eval_seed", "matchup", "captured", "steps", "pushes", "crosses"
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
