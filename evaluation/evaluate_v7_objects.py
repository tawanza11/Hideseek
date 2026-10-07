"""Paired object ablation on V7 validation maps with identical spawn cells."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from env.hide_seek_v7_env import HideSeekV7Env, ObjectLayout
from env.v7_maps import TRAIN_LAYOUTS, VALIDATION_LAYOUTS, layout_hash
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v7_hider import run_game as run_hider
from evaluation.evaluate_v7_seeker import run_game as run_seeker
from training.train_v2 import model_policy


MODES = {
    "full": (True, True),
    "no_block": (False, True),
    "no_ramp": (True, False),
    "no_objects": (False, False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare V7 object rules by paired games")
    parser.add_argument("--training-seed", type=int, choices=(41, 42, 43), default=41)
    parser.add_argument("--episodes-per-map", type=int, default=100)
    parser.add_argument("--seeker-model-dir", type=Path, required=True)
    parser.add_argument("--hider-model-dir", type=Path, required=True)
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes_per_map < 1:
        parser.error("--episodes-per-map must be positive")
    return args


def original_spawn(layout: ObjectLayout, seed: int) -> dict[str, tuple[int, int]]:
    env = HideSeekV7Env(layouts=(layout,))
    try:
        env.reset(seed=seed)
        return {"seeker_pos": env.seeker_pos, "hider_pos": env.hider_pos}
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    seed = args.training_seed
    seeker_path = args.seeker_model_dir / f"seeker_v7_route_seed{seed}.zip"
    hider_path = args.hider_model_dir / f"hider_v7_seed{seed}.zip"
    for path, role in ((seeker_path, "seeker"), (hider_path, "hider")):
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if ((metadata.get("version"), metadata.get("role"), metadata.get("seed")) != (7, role, seed)
                or metadata.get("training_layout_hash") != layout_hash(TRAIN_LAYOUTS)):
            raise ValueError(f"V7 model metadata mismatch: {path}")
    seeker = PPO.load(str(seeker_path), device="cpu")
    hider = PPO.load(str(hider_path), device="cpu")
    if seeker.observation_space.shape != (516,) or hider.observation_space.shape != (508,):
        raise ValueError("V7 object ablation requires route Seeker and 508-value Hider")
    v3_hider, _ = load_model(
        args.v3_model_dir / f"hider_v3_seed{seed}.zip", role="hider", stage="final", seed=seed,
    )
    v3_policy = model_policy(v3_hider)
    rows: list[dict[str, str | int]] = []
    for map_index, (name, layout) in enumerate(VALIDATION_LAYOUTS.items()):
        for episode in range(args.episodes_per_map):
            game_seed = 1_800_000 + map_index * 10_000 + episode
            spawn = original_spawn(layout, game_seed)
            for mode, (blocks, ramp) in MODES.items():
                seeker_result = run_seeker(
                    layout, game_seed, lambda observation: v3_policy(observation[:105]), "learned", seeker,
                    enable_blocks=blocks, enable_ramp=ramp, spawn=spawn,
                )
                hider_result = run_hider(
                    layout, game_seed, "teacher", "v7", v3_hider, hider,
                    enable_blocks=blocks, enable_ramp=ramp, spawn=spawn,
                )
                rows.extend((
                    {"map": name, "eval_seed": game_seed, "training_seed": seed,
                     "role": "seeker", "mode": mode, "success": seeker_result["captured"],
                     "steps": seeker_result["steps"], "pushes": seeker_result["pushes"],
                     "crosses": seeker_result["crosses"]},
                    {"map": name, "eval_seed": game_seed, "training_seed": seed,
                     "role": "hider", "mode": mode, "success": hider_result["survived"],
                     "steps": hider_result["steps"], "pushes": hider_result["pushes"],
                     "crosses": hider_result["crosses"]},
                ))
        print(f"Ablation {name} complete", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "map", "eval_seed", "training_seed", "role", "mode", "success", "steps", "pushes", "crosses"
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
