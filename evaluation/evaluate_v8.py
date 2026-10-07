"""Paired V8 object challenge validation with stochastic learned actions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from env.hide_seek_v7_env import HideSeekV7Env
from env.v8_maps import TRAIN_LAYOUTS, VALIDATION_LAYOUTS, ChallengeLayout, layout_hash
from training.v7_observation import with_route_hints
from training.v7_teacher import legal_actions, search_action
from training.v8_barricade import demonstration_policy


MODES = {
    "full": (True, True),
    "no_block": (False, True),
    "no_ramp": (True, False),
    "no_objects": (False, False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V8 on its unseen validation challenges")
    parser.add_argument("--training-seed", type=int, choices=(41, 42, 43), required=True)
    parser.add_argument("--episodes-per-map", type=int, default=100)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes_per_map < 1:
        parser.error("--episodes-per-map must be positive")
    return args


def load_policy(model_dir: Path, role: str, seed: int) -> PPO:
    path = model_dir / f"{role}_v8_seed{seed}.zip"
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        metadata.get("version"), metadata.get("role"), metadata.get("seed"),
        metadata.get("training_layout_hash"), metadata.get("max_steps"),
    ) != (8, role, seed, layout_hash(TRAIN_LAYOUTS), 27):
        raise ValueError(f"V8 model metadata mismatch: {path}")
    model = PPO.load(str(path), device="cpu")
    if model.observation_space.shape != ((516,) if role == "seeker" else (508,)):
        raise ValueError(f"V8 observation mismatch: {path}")
    return model


def choose_action(
    observation: np.ndarray, model: PPO | None, rng: np.random.Generator,
) -> int:
    allowed = legal_actions(observation)
    if not allowed:
        return 0
    if model is None:
        return int(rng.choice(allowed))
    model_observation = (
        with_route_hints(observation)
        if model.observation_space.shape == (516,) else observation
    )
    with torch.no_grad():
        tensor = torch.as_tensor(model_observation[None, :], device=model.device)
        probabilities = model.policy.get_distribution(tensor).distribution.probs[0].cpu().numpy()
    weights = np.asarray([probabilities[action] for action in allowed], dtype=np.float64)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        weights = np.full(len(allowed), 1 / len(allowed), dtype=np.float64)
    else:
        weights /= total
    return int(rng.choice(allowed, p=weights))


def run_game(
    challenge: ChallengeLayout, eval_seed: int, role: str,
    model: PPO | None, mode: str,
) -> dict[str, int]:
    blocks, ramp = MODES[mode]
    env = HideSeekV7Env(
        layouts=(challenge.arena,), role=role, max_steps=27,
        opponent_policy=demonstration_policy(eval_seed + 17)
        if role == "seeker" else search_action,
        enable_blocks=blocks, enable_ramp=ramp,
    )
    observation, _ = env.reset(seed=eval_seed, options={
        "seeker_pos": challenge.seeker_spawn,
        "hider_pos": challenge.hider_spawn,
    })
    rng = np.random.default_rng(eval_seed + 29)
    pushes = crosses = 0
    try:
        for step in range(1, env.max_steps + 1):
            action = choose_action(observation, model, rng)
            observation, _, terminated, truncated, info = env.step(action)
            pushes += int((role, "block_pushed") in env.last_events)
            crosses += int((role, "climbed") in env.last_events)
            if terminated or truncated:
                success = bool(info["captured"]) if role == "seeker" else bool(truncated)
                return {"success": int(success), "steps": step,
                        "pushes": pushes, "crosses": crosses}
        raise RuntimeError("V8 game did not terminate")
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    models = {role: load_policy(args.model_dir, role, args.training_seed)
              for role in ("seeker", "hider")}
    rows: list[dict[str, str | int]] = []
    variants = (
        ("seeker", "full", "learned"), ("seeker", "full", "random"),
        ("seeker", "no_ramp", "learned"), ("seeker", "no_block", "learned"),
        ("seeker", "no_objects", "learned"),
        ("hider", "full", "learned"), ("hider", "full", "random"),
        ("hider", "no_block", "learned"), ("hider", "no_ramp", "learned"),
        ("hider", "no_objects", "learned"),
    )
    for map_index, (name, challenge) in enumerate(VALIDATION_LAYOUTS.items()):
        counts = {role: 0 for role in ("seeker", "hider")}
        for episode in range(args.episodes_per_map):
            eval_seed = 3_000_000 + map_index * 10_000 + episode
            for role, mode, policy in variants:
                result = run_game(
                    challenge, eval_seed, role,
                    models[role] if policy == "learned" else None, mode,
                )
                rows.append({"map": name, "training_seed": args.training_seed,
                             "eval_seed": eval_seed, "role": role, "mode": mode,
                             "policy": policy, **result})
                if mode == "full" and policy == "learned":
                    counts[role] += result["success"]
        print(f"{name} full learned success: {counts}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "map", "training_seed", "eval_seed", "role", "mode", "policy",
            "success", "steps", "pushes", "crosses",
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
