"""Open V8 final challenges once, only after validation artifacts pass and lock."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from env.v8_maps import FINAL_LAYOUTS, layout_hash
from evaluation.evaluate_v8 import load_policy, run_game
from evaluation.select_v8 import EPISODES, SEEDS, select


EXPECTED_FINAL_HASH = "39a228af09a010bef8c0750fd3ceac5716cf42cd9c13126bc4e746a1be36ce5e"
VARIANTS = (
    ("seeker", "full", "learned"), ("seeker", "full", "random"),
    ("seeker", "no_ramp", "learned"), ("seeker", "no_block", "learned"),
    ("seeker", "no_objects", "learned"),
    ("hider", "full", "learned"), ("hider", "full", "random"),
    ("hider", "no_block", "learned"), ("hider", "no_ramp", "learned"),
    ("hider", "no_objects", "learned"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate locked V8 policies on held-out challenges")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--v7-model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def verify_selection(results_dir: Path, model_dir: Path, v7_model_dir: Path) -> None:
    locked = json.loads((results_dir / "v8_selection.json").read_text(encoding="utf-8"))
    current = select(results_dir, model_dir, v7_model_dir)
    if locked != current:
        raise ValueError("V8 validation evidence or model artifacts changed after selection")
    if not locked["eligible_for_final"]:
        raise ValueError("V8 failed validation; final challenges remain unopened")
    if layout_hash(FINAL_LAYOUTS) != EXPECTED_FINAL_HASH:
        raise ValueError("V8 final layouts changed")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    verify_selection(args.results_dir, args.model_dir, args.v7_model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempt = args.output_dir / "v8_final_attempt.json"
    output = args.output_dir / "v8_final.csv"
    pending = args.output_dir / "v8_final.pending.csv"
    if any(path.exists() for path in (attempt, output, pending)):
        raise FileExistsError("V8 final evaluation was already opened")
    models = {
        (role, seed): load_policy(args.model_dir, role, seed)
        for seed in SEEDS for role in ("seeker", "hider")
    }
    attempt.write_text(json.dumps({
        "status": "started", "final_layout_hash": EXPECTED_FINAL_HASH,
        "selection": str(args.results_dir / "v8_selection.json"),
    }, indent=2) + "\n", encoding="utf-8")
    with pending.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "map", "training_seed", "eval_seed", "role", "mode", "policy",
            "success", "steps", "pushes", "crosses",
        ))
        writer.writeheader()
        for map_index, (name, challenge) in enumerate(FINAL_LAYOUTS.items()):
            for seed in SEEDS:
                for episode in range(EPISODES):
                    eval_seed = 4_000_000 + map_index * 10_000 + episode
                    for role, mode, policy in VARIANTS:
                        result = run_game(
                            challenge, eval_seed, role,
                            models[(role, seed)] if policy == "learned" else None,
                            mode,
                        )
                        writer.writerow({
                            "map": name, "training_seed": seed,
                            "eval_seed": eval_seed, "role": role,
                            "mode": mode, "policy": policy, **result,
                        })
            file.flush()
            print(f"V8 final {name} complete", flush=True)
    pending.replace(output)
    attempt.write_text(json.dumps({
        "status": "complete", "final_layout_hash": EXPECTED_FINAL_HASH,
        "selection": str(args.results_dir / "v8_selection.json"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Final results: {output}")


if __name__ == "__main__":
    main()
