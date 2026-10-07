"""Continue a V3 Hider in the object-aware V7 game against varied Seekers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from env.hide_seek_v7_env import HideSeekV7Env
from env.v7_maps import TRAIN_LAYOUTS, layout_hash
from evaluation.evaluate_v3 import load_model
from training.v7_teacher import search_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a V7 Hider on object layouts")
    parser.add_argument("--seed", type=int, choices=(41, 42, 43), default=41)
    parser.add_argument("--ppo-steps", type=int, default=150_000)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.ppo_steps < 1:
        parser.error("--ppo-steps must be positive")
    return args


def extend_v3_weights(target: PPO, source: PPO) -> None:
    """Retain the original 105 observation weights; initialize new channels at zero."""
    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    for name, source_value in source_state.items():
        target_value = target_state[name]
        if target_value.shape == source_value.shape:
            target_state[name] = source_value.clone()
        elif name in (
            "mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"
        ) and target_value.shape[0] == source_value.shape[0] and target_value.shape[1] > source_value.shape[1]:
            expanded = torch.zeros_like(target_value)
            expanded[:, : source_value.shape[1]] = source_value
            target_state[name] = expanded
        else:
            raise ValueError(f"Cannot transfer V3 weight {name}: {source_value.shape} -> {target_value.shape}")
    target.policy.load_state_dict(target_state)


def main() -> None:
    args = parse_args()
    source_path = args.model_dir / f"hider_v3_seed{args.seed}.zip"
    source, max_steps = load_model(source_path, role="hider", stage="final", seed=args.seed)
    env = HideSeekV7Env(
        layouts=tuple(TRAIN_LAYOUTS.values()), role="hider", max_steps=max_steps,
        opponent_policies=(None, search_action),
    )
    env.reset(seed=args.seed + 7_000)
    model = PPO(
        "MlpPolicy", env, seed=args.seed + 7_000,
        n_steps=512, batch_size=64, verbose=0, device="cpu",
    )
    extend_v3_weights(model, source)
    try:
        model.learn(total_timesteps=args.ppo_steps)
        model_path = args.output_dir / "models" / f"hider_v7_seed{args.seed}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        model_path.with_suffix(".json").write_text(json.dumps({
            "version": 7,
            "role": "hider",
            "seed": args.seed,
            "source_model": str(source_path),
            "source_timesteps": source.num_timesteps,
            "ppo_requested_steps": args.ppo_steps,
            "ppo_actual_steps": model.num_timesteps,
            "opponent_pool": ["random", "observable_search_teacher"],
            "training_layouts": list(TRAIN_LAYOUTS),
            "training_layout_hash": layout_hash(TRAIN_LAYOUTS),
            "observation_size": env.observation_space.shape[0],
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Model: {model_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
