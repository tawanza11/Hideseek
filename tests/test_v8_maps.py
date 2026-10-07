"""Structural checks only; no V8 final-map episode is played."""

import unittest

from env.hide_seek_v3_env import DEFAULT_MAP
from env.hide_seek_v7_env import ObjectLayout
from env.v4_maps import TEST_MAPS as V4_TEST_MAPS
from env.v4_maps import TRAIN_MAPS as V4_TRAIN_MAPS
from env.v4_maps import VALIDATION_MAPS as V4_VALIDATION_MAPS
from env.v5_maps import FINAL_TEST_MAPS as V5_FINAL_TEST_MAPS
from env.v5_maps import TRAIN_MAPS as V5_TRAIN_MAPS
from env.v5_maps import VALIDATION_MAPS as V5_VALIDATION_MAPS
from env.v7_maps import FINAL_LAYOUTS as V7_FINAL_LAYOUTS
from env.v7_maps import TRAIN_LAYOUTS as V7_TRAIN_LAYOUTS
from env.v7_maps import VALIDATION_LAYOUTS as V7_VALIDATION_LAYOUTS
from env.v8_maps import (
    CANONICAL_HIDER_SPAWN,
    CANONICAL_ROWS,
    CANONICAL_SEEKER_SPAWN,
    FINAL_LAYOUTS,
    FINAL_SEED_START,
    TRAIN_LAYOUTS,
    TRAIN_SEED_START,
    VALIDATION_LAYOUTS,
    VALIDATION_SEED_START,
    ChallengeLayout,
    layout_hash,
    make_challenge,
)


EXPECTED_HASHES = {
    "V8T": "985410095794e0b85f66bfab01a002a04a32436566a879cd825f2c33be9f9c93",
    "V8E": "af8cda1a7968e5df7beb3f223df061b2b22715e1ad288e298dde42c0bd81b0b1",
    "V8F": "39a228af09a010bef8c0750fd3ceac5716cf42cd9c13126bc4e746a1be36ce5e",
}


class V8MapTests(unittest.TestCase):
    def test_canonical_challenge_is_first_training_layout(self) -> None:
        challenge = TRAIN_LAYOUTS["V8T001"]
        self.assertEqual(challenge.arena.map_rows, CANONICAL_ROWS)
        self.assertEqual(challenge.arena.block, (5, 3))
        self.assertEqual(challenge.arena.ramp, (6, 7))
        self.assertEqual(challenge.arena.low_wall, (5, 7))
        self.assertEqual(challenge.seeker_spawn, CANONICAL_SEEKER_SPAWN)
        self.assertEqual(challenge.hider_spawn, CANONICAL_HIDER_SPAWN)
        self.assertEqual(challenge.validate(), 10)

    def test_splits_are_distinct_and_every_layout_keeps_the_challenge(self) -> None:
        groups = (TRAIN_LAYOUTS, VALIDATION_LAYOUTS, FINAL_LAYOUTS)
        self.assertEqual([len(group) for group in groups], [100, 10, 20])
        self.assertEqual(list(TRAIN_LAYOUTS)[0], "V8T001")
        self.assertEqual(list(VALIDATION_LAYOUTS)[0], "V8E001")
        self.assertEqual(list(FINAL_LAYOUTS)[0], "V8F001")
        rows = [challenge.arena.map_rows for group in groups for challenge in group.values()]
        self.assertEqual(len(rows), len(set(rows)))
        prior_rows = {
            DEFAULT_MAP,
            *V4_TRAIN_MAPS.values(), *V4_VALIDATION_MAPS.values(), *V4_TEST_MAPS.values(),
            *V5_TRAIN_MAPS.values(), *V5_VALIDATION_MAPS.values(), *V5_FINAL_TEST_MAPS.values(),
        }
        prior_rows.update(
            layout.map_rows
            for group in (V7_TRAIN_LAYOUTS, V7_VALIDATION_LAYOUTS, V7_FINAL_LAYOUTS)
            for layout in group.values()
        )
        self.assertTrue(set(rows).isdisjoint(prior_rows))
        for group in groups:
            for challenge in group.values():
                self.assertEqual(challenge.validate(), 10)

    def test_generation_and_fingerprints_are_deterministic(self) -> None:
        for seed in (TRAIN_SEED_START, TRAIN_SEED_START + 1,
                     VALIDATION_SEED_START, FINAL_SEED_START):
            self.assertEqual(make_challenge(seed), make_challenge(seed))
        for prefix, group in (("V8T", TRAIN_LAYOUTS), ("V8E", VALIDATION_LAYOUTS),
                              ("V8F", FINAL_LAYOUTS)):
            self.assertEqual(layout_hash(group), EXPECTED_HASHES[prefix])
            self.assertEqual(layout_hash(dict(reversed(list(group.items())))), EXPECTED_HASHES[prefix])

    def test_reopening_doorway_from_seeker_side_is_rejected(self) -> None:
        rows = list(CANONICAL_ROWS)
        rows[4] = rows[4][:4] + "." + rows[4][5:]
        invalid = ChallengeLayout(
            arena=ObjectLayout(tuple(rows), block=(5, 3), ramp=(6, 7), low_wall=(5, 7)),
            seeker_spawn=CANONICAL_SEEKER_SPAWN,
            hider_spawn=CANONICAL_HIDER_SPAWN,
        )
        with self.assertRaises(ValueError):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
