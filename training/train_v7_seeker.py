"""Train a V7 Seeker: observable search imitation followed by PPO games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from env.hide_seek_v7_env import HideSeekV7Env
from env.v7_maps import TRAIN_LAYOUTS, layout_hash
from evaluation.evaluate_v3 import load_model
from training.train_v2 import model_policy
from training.v7_observation import SeekerPlannerObservation
from training.v7_teacher import search_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a V7 Seeker on object layouts")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--teacher-steps", type=int, default=30_000)
    parser.add_argument("--imitation-epochs", type=int, default=8)
    parser.add_argument("--ppo-steps", type=int, default=100_000)
    parser.add_argument("--route-hints", action="store_true")
    parser.add_argument("--ppo-hider-pool", action="store_true")
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-5)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.seed < 0 or args.teacher_steps < 1 or args.imitation_epochs < 1 or args.ppo_steps < 0:
        parser.error("seed must be nonnegative; teacher steps/epochs positive; PPO steps nonnegative")
    if not 0 < args.learning_rate <= 1e-3:
        parser.error("--learning-rate must be in (0, 0.001]")
    if not 0 < args.ppo_learning_rate <= 1e-3:
        parser.error("--ppo-learning-rate must be in (0, 0.001]")
    return args


def collect_teacher_data(env, steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    states = np.empty((steps, env.observation_space.shape[0]), dtype=np.float32)
    actions = np.empty(steps, dtype=np.int64)
    observation, _ = env.reset(seed=seed)
    for index in range(steps):
        action = search_action(observation[:508])
        states[index] = observation
        actions[index] = action
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
    return states, actions


def imitate(model: PPO, states: np.ndarray, actions: np.ndarray, epochs: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    policy = model.policy
    policy.train()
    losses: list[float] = []
    for epoch in range(epochs):
        order = rng.permutation(len(states))
        batch_losses = []
        for start in range(0, len(order), 256):
            indices = order[start : start + 256]
            observations = torch.as_tensor(states[indices], device=policy.device)
            selected = torch.as_tensor(actions[indices], device=policy.device)
            distribution = policy.get_distribution(observations)
            loss = -distribution.log_prob(selected).mean()
            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(batch_losses))
        losses.append(epoch_loss)
        print(f"Imitation epoch {epoch + 1}/{epochs}: loss {epoch_loss:.4f}", flush=True)
    policy.eval()
    return losses


def main() -> None:
    args = parse_args()
    base_env = HideSeekV7Env(
        layouts=tuple(TRAIN_LAYOUTS.values()), role="seeker", novelty_bonus=0.01,
    )
    env = SeekerPlannerObservation(base_env) if args.route_hints else base_env
    model = PPO(
        "MlpPolicy", env, seed=args.seed, n_steps=512, batch_size=64,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        learning_rate=args.learning_rate, verbose=0, device="cpu",
    )
    ppo_env = None
    try:
        states, actions = collect_teacher_data(env, args.teacher_steps, args.seed + 7_000)
        losses = imitate(model, states, actions, args.imitation_epochs, args.seed)
        if args.ppo_steps:
            model.lr_schedule = lambda _: args.ppo_learning_rate
            for group in model.policy.optimizer.param_groups:
                group["lr"] = args.ppo_learning_rate
            if args.ppo_hider_pool:
                hider_paths = [args.v3_model_dir / f"hider_v3_seed{seed}.zip" for seed in (41, 42, 43)]
                hiders = [load_model(path, role="hider", stage="final", seed=seed)[0]
                          for seed, path in zip((41, 42, 43), hider_paths, strict=True)]
                policies = [model_policy(hider) for hider in hiders]
                ppo_base_env = HideSeekV7Env(
                    layouts=tuple(TRAIN_LAYOUTS.values()), role="seeker", novelty_bonus=0.01,
                    opponent_policies=(None, *(lambda obs, policy=policy: policy(obs[:105])
                                               for policy in policies)),
                )
                ppo_env = SeekerPlannerObservation(ppo_base_env) if args.route_hints else ppo_base_env
                model.set_env(ppo_env)
            print(f"PPO: {args.ppo_steps} requested steps", flush=True)
            model.learn(total_timesteps=args.ppo_steps)
        prefix = "seeker_v7_route" if args.route_hints else "seeker_v7"
        model_path = args.output_dir / "models" / f"{prefix}_seed{args.seed}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        model_path.with_suffix(".json").write_text(json.dumps({
            "version": 7,
            "role": "seeker",
            "seed": args.seed,
            "algorithm": "PPO after observable-state search imitation",
            "route_hints": args.route_hints,
            "teacher_steps": args.teacher_steps,
            "imitation_epochs": args.imitation_epochs,
            "imitation_losses": losses,
            "ppo_requested_steps": args.ppo_steps,
            "ppo_actual_steps": model.num_timesteps,
            "ppo_hider_pool": args.ppo_hider_pool,
            "learning_rate": args.learning_rate,
            "ppo_learning_rate": args.ppo_learning_rate,
            "training_layouts": list(TRAIN_LAYOUTS),
            "training_layout_hash": layout_hash(TRAIN_LAYOUTS),
            "observation_size": env.observation_space.shape[0],
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Model: {model_path}", flush=True)
    finally:
        env.close()
        if ppo_env is not None:
            ppo_env.close()


if __name__ == "__main__":
    main()
