"""Continue both V7 policies on barricade-and-ramp challenge layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from env.hide_seek_v7_env import HideSeekV7Env
from env.v8_maps import TRAIN_LAYOUTS, ChallengeLayout, layout_hash
from evaluation.evaluate_v7_seeker import learned_action
from training.train_v7_seeker import imitate
from training.v7_observation import SeekerPlannerObservation
from training.v7_teacher import search_action
from training.v8_barricade import barricade_direction, demonstration_policy


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V8 object strategies")
    parser.add_argument("--role", choices=("seeker", "hider"), required=True)
    parser.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    parser.add_argument("--imitation-steps", type=int, default=20_000)
    parser.add_argument("--imitation-epochs", type=int, default=5)
    parser.add_argument("--ppo-steps", type=int, default=100_000)
    parser.add_argument("--v7-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    if min(args.imitation_steps, args.imitation_epochs, args.ppo_steps) < 1:
        parser.error("imitation steps/epochs and PPO steps must be positive")
    return args


def spawn(layout: ChallengeLayout) -> dict[str, tuple[int, int]]:
    return {"seeker_pos": layout.seeker_spawn, "hider_pos": layout.hider_spawn}


class ObjectReward(gym.Wrapper):
    def __init__(self, env: HideSeekV7Env, layouts: tuple[ChallengeLayout, ...], role: str):
        super().__init__(env)
        self.layouts = layouts
        self.role = role

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.role == "seeker" and ("seeker", "climbed") in self.env.last_events:
            reward += 0.1
        if self.role == "hider" and ("hider", "block_pushed") in self.env.last_events:
            layout = self.layouts[self.env.layout_index]
            dx = layout.arena.block[0] - layout.hider_spawn[0]
            dy = layout.arena.block[1] - layout.hider_spawn[1]
            target = (layout.arena.block[0] + dx, layout.arena.block[1] + dy)
            if target in self.env.blocks:
                reward += 0.3
        return observation, reward, terminated, truncated, info


def collect_hider_data(
    env: HideSeekV7Env, model: PPO, count: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.empty((count, 508), dtype=np.float32)
    actions = np.empty(count, dtype=np.int64)
    observation, _ = env.reset(seed=seed)
    for index in range(count):
        push = barricade_direction(observation)
        action = push if push is not None else learned_action(model, observation)
        states[index] = observation
        actions[index] = action
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
    return states, actions


def collect_seeker_data(
    env: SeekerPlannerObservation, count: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.empty((count, 516), dtype=np.float32)
    actions = np.empty(count, dtype=np.int64)
    observation, _ = env.reset(seed=seed)
    for index in range(count):
        action = search_action(observation[:508])
        states[index] = observation
        actions[index] = action
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
    return states, actions


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    challenges = tuple(TRAIN_LAYOUTS.values())
    base = HideSeekV7Env(
        layouts=tuple(item.arena for item in challenges),
        spawns_by_layout=tuple(spawn(item) for item in challenges),
        role=args.role, max_steps=27,
        opponent_policy=demonstration_policy(args.seed + 8_000)
        if args.role == "seeker" else None,
        opponent_policies=(None, search_action)
        if args.role == "hider" else None,
    )
    shaped = ObjectReward(base, challenges, args.role)
    env = SeekerPlannerObservation(shaped) if args.role == "seeker" else shaped
    source_path = args.v7_model_dir / (
        f"seeker_v7_route_seed{args.seed}.zip" if args.role == "seeker"
        else f"hider_v7_seed{args.seed}.zip"
    )
    model = PPO.load(str(source_path), device="cpu")
    model.set_env(env)
    try:
        if args.role == "seeker":
            states, actions = collect_seeker_data(env, args.imitation_steps, args.seed + 9_000)
        else:
            states, actions = collect_hider_data(base, model, args.imitation_steps, args.seed + 9_000)
        losses = imitate(model, states, actions, args.imitation_epochs, args.seed)
        model.learn(total_timesteps=args.ppo_steps, reset_num_timesteps=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"{args.role}_v8_seed{args.seed}.zip"
        model.save(str(path))
        path.with_suffix(".json").write_text(json.dumps({
            "version": 8, "role": args.role, "seed": args.seed,
            "source_model": source_path.name, "source_hash": file_hash(source_path),
            "training_layout_hash": layout_hash(TRAIN_LAYOUTS),
            "max_steps": 27, "imitation_steps": args.imitation_steps,
            "imitation_epochs": args.imitation_epochs, "imitation_losses": losses,
            "ppo_requested_steps": args.ppo_steps,
            "ppo_actual_steps": model.num_timesteps,
            "observation_size": env.observation_space.shape[0],
            "object_reward": 0.1 if args.role == "seeker" else 0.3,
            "opponent": "barricade_demonstration" if args.role == "seeker"
            else "random_or_observable_search",
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Model: {path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
