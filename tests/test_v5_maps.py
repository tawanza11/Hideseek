"""Structural and split-isolation checks for the V5 procedural maps."""

import unittest

from env.hide_seek_v3_env import DEFAULT_MAP, HideSeekV3Env
from env.v4_maps import TEST_MAPS as V4_TEST_MAPS
from env.v4_maps import TRAIN_MAPS as V4_TRAIN_MAPS
from env.v4_maps import VALIDATION_MAPS as V4_VALIDATION_MAPS
from env.v5_maps import FINAL_TEST_MAPS, TRAIN_MAPS, VALIDATION_MAPS, generate_map
from training.train_v5 import TRAINING_MAPS, TRAINING_MAP_NAMES


class V5MapTests(unittest.TestCase):
    def test_fixed_seed_generation_is_deterministic(self) -> None:
        for seed in (500_000, 500_017, 600_003, 700_011):
            with self.subTest(seed=seed):
                self.assertEqual(generate_map(seed), generate_map(seed))

    def test_splits_have_expected_sizes_and_no_legacy_overlap(self) -> None:
        legacy_maps = {
            DEFAULT_MAP,
            *V4_TRAIN_MAPS.values(),
            *V4_VALIDATION_MAPS.values(),
            *V4_TEST_MAPS.values(),
        }
        v5_maps = (
            *TRAIN_MAPS.values(),
            *VALIDATION_MAPS.values(),
            *FINAL_TEST_MAPS.values(),
        )
        self.assertEqual(len(TRAIN_MAPS), 100)
        self.assertEqual(len(VALIDATION_MAPS), 10)
        self.assertEqual(len(FINAL_TEST_MAPS), 20)
        self.assertEqual(len(v5_maps), len(set(v5_maps)))
        self.assertTrue(legacy_maps.isdisjoint(v5_maps))

    def test_every_map_is_a_valid_connected_10x10_grid(self) -> None:
        maps = (
            *TRAIN_MAPS.values(),
            *VALIDATION_MAPS.values(),
            *FINAL_TEST_MAPS.values(),
        )
        for rows in maps:
            with self.subTest(map_rows=rows):
                self.assertEqual(len(rows), 10)
                self.assertTrue(all(len(row) == 10 for row in rows))
                self.assertTrue(all(set(row) <= {".", "#"} for row in rows))
                self.assertTrue(any("#" in row[1:-1] for row in rows[1:-1]))
                HideSeekV3Env(map_rows=rows)

    def test_training_pool_excludes_new_validation_and_final_maps(self) -> None:
        self.assertEqual(len(TRAINING_MAPS), len(TRAINING_MAP_NAMES))
        self.assertEqual(len(set(TRAINING_MAPS)), 108)
        self.assertTrue(set(TRAINING_MAPS).isdisjoint(VALIDATION_MAPS.values()))
        self.assertTrue(set(TRAINING_MAPS).isdisjoint(FINAL_TEST_MAPS.values()))


if __name__ == "__main__":
    unittest.main()
