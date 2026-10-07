"""One guarded held-out V7 evaluation after validation selection is locked."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from env.v7_maps import FINAL_LAYOUTS, layout_hash
from evaluation.evaluate_v3 import load_model
from evaluation.evaluate_v7_hider import run_game as run_hider
from evaluation.evaluate_v7_seeker import learned_action, run_game as run_seeker
from evaluation.select_v7 import EPISODES, SEEDS, select
from training.train_v2 import model_policy


EXPECTED_FINAL_HASH = "58d57fa565eed14bb6c4bd213441af09e1cbfcf3b642f1a6d27bd18c460cbc5b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate locked V7 models on final layouts once")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v3-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def verify_selection(results_dir: Path, model_dir: Path, v3_model_dir: Path) -> None:
    selection_path = results_dir / "v7_selection.json"
    locked = json.loads(selection_path.read_text(encoding="utf-8"))
    current = select(results_dir, model_dir, v3_model_dir)
    if locked != current:
        raise ValueError("V7 models or validation evidence changed after selection lock")
    if not locked["eligible_for_final"]:
        raise ValueError("No V7 model set passed validation; final maps remain unopened")
    if layout_hash(FINAL_LAYOUTS) != EXPECTED_FINAL_HASH:
        raise ValueError("V7 final layout hash changed")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    verify_selection(args.results_dir, args.model_dir, args.v3_model_dir)
    seeker_output = args.output_dir / "v7_final_seeker.csv"
    hider_output = args.output_dir / "v7_final_hider.csv"
    attempt_path = args.output_dir / "v7_final_attempt.json"
    if seeker_output.exists() or hider_output.exists() or attempt_path.exists():
        raise FileExistsError("V7 final results already exist; keep the first opening intact")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeker_pending = args.output_dir / "v7_final_seeker.pending.csv"
    hider_pending = args.output_dir / "v7_final_hider.pending.csv"
    if seeker_pending.exists() or hider_pending.exists():
        raise FileExistsError("Previous V7 final evaluation is incomplete")
    seekers = {
        seed: PPO.load(str(args.model_dir / f"seeker_v7_route_seed{seed}.zip"), device="cpu")
        for seed in SEEDS
    }
    hiders = {
        seed: PPO.load(str(args.model_dir / f"hider_v7_seed{seed}.zip"), device="cpu")
        for seed in SEEDS
    }
    old_hiders = {
        seed: load_model(
            args.v3_model_dir / f"hider_v3_seed{seed}.zip",
            role="hider", stage="final", seed=seed,
        )[0]
        for seed in SEEDS
    }
    attempt_path.write_text(json.dumps({
        "status": "started", "selection": str(args.results_dir / "v7_selection.json"),
        "final_layout_hash": EXPECTED_FINAL_HASH,
    }, indent=2) + "\n", encoding="utf-8")
    with seeker_pending.open("w", newline="", encoding="utf-8") as seeker_file, \
            hider_pending.open("w", newline="", encoding="utf-8") as hider_file:
        seeker_writer = csv.DictWriter(seeker_file, fieldnames=(
            "map", "training_seed", "eval_seed", "hider", "seeker", "captured", "steps", "pushes", "crosses"
        ))
        hider_writer = csv.DictWriter(hider_file, fieldnames=(
            "map", "training_seed", "eval_seed", "seeker", "hider", "survived", "steps", "pushes", "crosses"
        ))
        seeker_writer.writeheader()
        hider_writer.writeheader()
        for map_index, (name, layout) in enumerate(FINAL_LAYOUTS.items()):
            for seed in SEEDS:
                old_policy = model_policy(old_hiders[seed])
                opponent_policies = {
                    "v3": lambda observation, policy=old_policy: policy(observation[:105]),
                    "v7": lambda observation, policy=hiders[seed]: learned_action(policy, observation),
                }
                for episode in range(EPISODES):
                    eval_seed = 2_000_000 + map_index * 10_000 + episode
                    for hider_name, opponent in opponent_policies.items():
                        for seeker_name in ("random", "teacher", "learned"):
                            result = run_seeker(
                                layout, eval_seed, opponent, seeker_name, seekers[seed]
                            )
                            seeker_writer.writerow({
                                "map": name, "training_seed": seed, "eval_seed": eval_seed,
                                "hider": hider_name, "seeker": seeker_name, **result,
                            })
                    for seeker_name in ("random", "teacher"):
                        for hider_name in ("random", "v3", "v7"):
                            result = run_hider(
                                layout, eval_seed, seeker_name, hider_name,
                                old_hiders[seed], hiders[seed],
                            )
                            hider_writer.writerow({
                                "map": name, "training_seed": seed, "eval_seed": eval_seed,
                                "seeker": seeker_name, "hider": hider_name, **result,
                            })
            seeker_file.flush()
            hider_file.flush()
            print(f"Final map {name} complete", flush=True)
    seeker_pending.replace(seeker_output)
    hider_pending.replace(hider_output)
    attempt_path.write_text(json.dumps({
        "status": "complete", "selection": str(args.results_dir / "v7_selection.json"),
        "final_layout_hash": EXPECTED_FINAL_HASH,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Final results: {seeker_output} and {hider_output}")


if __name__ == "__main__":
    main()
