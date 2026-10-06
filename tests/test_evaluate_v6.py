"""V6 final evaluation stays locked to the validated results and models."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import evaluate_v6


class V6SelectionTests(unittest.TestCase):
    def write_results(self, output_dir: Path, partition: str, map_names: list[str]) -> None:
        path = evaluate_v6.result_path(output_dir, partition)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=(
                "map", "training_seed", "eval_seed", "matchup", "captured",
            ), lineterminator="\n")
            writer.writeheader()
            for map_index, map_name in enumerate(map_names):
                for seed in evaluate_v6.EVAL_SEEDS:
                    for matchup in evaluate_v6.MATCHUPS:
                        caught = (
                            map_index < 6 if matchup == "V6 curriculum"
                            else map_index < 5
                        )
                        writer.writerow({
                            "map": map_name,
                            "training_seed": seed,
                            "eval_seed": map_index * 10_000 + seed,
                            "matchup": matchup,
                            "captured": int(caught),
                        })

    def test_selection_lock_rejects_changed_models_or_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            model_dir = output_dir / "models"
            model_dir.mkdir()
            with patch.object(evaluate_v6, "EPISODES_PER_MAP", 1):
                self.write_results(
                    output_dir, "validation", list(evaluate_v6.VALIDATION_MAPS)
                )
                self.write_results(output_dir, "default", ["default"])
                for variant in ("flat", "curriculum"):
                    for seed in evaluate_v6.EVAL_SEEDS:
                        (model_dir / f"seeker_v6_{variant}_seed{seed}.zip").write_bytes(b"model")
                decision = evaluate_v6.lock_selection(output_dir, model_dir)
                self.assertEqual(decision["selection"], "curriculum")
                evaluate_v6.require_lock(output_dir, model_dir)
                with self.assertRaises(FileExistsError):
                    evaluate_v6.lock_selection(output_dir, model_dir)

                changed_model = model_dir / "seeker_v6_flat_seed41.zip"
                changed_model.write_bytes(b"different model")
                with self.assertRaisesRegex(ValueError, "models changed"):
                    evaluate_v6.require_lock(output_dir, model_dir)
                changed_model.write_bytes(b"model")
                self.write_results(output_dir, "default", ["renamed"])
                with self.assertRaisesRegex(ValueError, "default results changed"):
                    evaluate_v6.require_lock(output_dir, model_dir)


if __name__ == "__main__":
    unittest.main()
