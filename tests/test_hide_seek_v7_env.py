"""Object rules and observation boundaries for the V7 game."""

import unittest

import numpy as np

from env.hide_seek_v7_env import HideSeekV7Env, ObjectLayout
from env.v7_maps import FINAL_LAYOUTS, TRAIN_LAYOUTS, VALIDATION_LAYOUTS


ROWS = ("#####", "#...#", "#.#.#", "#...#", "#####")
LAYOUT = ObjectLayout(ROWS, block=(2, 1), ramp=(1, 2), low_wall=(2, 2))


class HideSeekV7EnvTests(unittest.TestCase):
    def test_push_and_ramp_change_movement_and_vision(self) -> None:
        env = HideSeekV7Env(layouts=(LAYOUT,), opponent_policy=lambda _: 0)
        observation, _ = env.reset(seed=8)
        self.assertTrue(env.observation_space.contains(observation))
        env.seeker_pos, env.hider_pos = (1, 1), (3, 3)

        env.step(3)  # Push the block right.
        self.assertEqual(env.seeker_pos, (2, 1))
        self.assertEqual(env.blocks, {(3, 1)})
        self.assertIn(("seeker", "block_pushed"), env.last_events)
        self.assertFalse(env._line_of_sight((2, 1), (3, 2)))
        self.assertEqual(env._move((3, 2), 0), (3, 2))  # Block against wall seals that entry.

        env.seeker_pos, env.hider_pos = (1, 2), (3, 3)
        env.step(3)  # From the ramp, cross the low wall in one move.
        self.assertEqual(env.seeker_pos, (3, 2))
        self.assertEqual(env.low_wall, (2, 2))
        self.assertIn(("seeker", "climbed"), env.last_events)

    def test_unpushable_block_and_no_ramp_cannot_cross(self) -> None:
        env = HideSeekV7Env(layouts=(LAYOUT,))
        env.reset(seed=1)
        env.seeker_pos, env.hider_pos = (1, 2), (3, 3)
        self.assertEqual(env._move((3, 2), 2), (3, 2))
        env.seeker_pos, env.hider_pos = (1, 1), (3, 1)
        self.assertEqual(env._move((1, 1), 3), (1, 1))
        self.assertEqual(env.blocks, {(2, 1)})

    def test_hidden_opponent_is_not_leaked_and_reset_restores_objects(self) -> None:
        env = HideSeekV7Env(layouts=(LAYOUT,))
        env.reset(seed=2)
        env.seeker_pos, env.hider_pos = (1, 2), (3, 2)
        observation = env._observation_for("seeker")
        np.testing.assert_array_equal(observation[2:5], [0, 0, 0])
        self.assertEqual(observation.shape, (133,))
        env.blocks = {(3, 1)}
        env.reset(seed=2)
        self.assertEqual(env.blocks, {(2, 1)})

    def test_new_map_partitions_are_disjoint(self) -> None:
        groups = [TRAIN_LAYOUTS, VALIDATION_LAYOUTS, FINAL_LAYOUTS]
        self.assertEqual([len(group) for group in groups], [100, 10, 20])
        maps = [layout.map_rows for group in groups for layout in group.values()]
        self.assertEqual(len(maps), len(set(maps)))
        self.assertTrue(all(layout.validate() == 10 for group in groups for layout in group.values()))

    def test_fixed_spawns_and_object_ablation(self) -> None:
        env = HideSeekV7Env(layouts=(LAYOUT,), enable_blocks=False, enable_ramp=False)
        observation, _ = env.reset(
            seed=3, options={"seeker_pos": (1, 2), "hider_pos": (3, 2)}
        )
        self.assertEqual((env.seeker_pos, env.hider_pos), ((1, 2), (3, 2)))
        self.assertEqual(env.blocks, set())
        np.testing.assert_array_equal(observation[30:105], np.zeros(75))
        self.assertEqual(env._move((1, 2), 3), (1, 2))
        with self.assertRaises(ValueError):
            env.reset(options={"seeker_pos": (2, 1), "hider_pos": (2, 1)})

    def test_layout_specific_spawns_follow_selected_layout(self) -> None:
        spawns = ({"seeker_pos": (1, 2), "hider_pos": (3, 2)},)
        env = HideSeekV7Env(layouts=(LAYOUT,), spawns_by_layout=spawns)
        for seed in (1, 2):
            env.reset(seed=seed)
            self.assertEqual((env.seeker_pos, env.hider_pos), ((1, 2), (3, 2)))
        with self.assertRaises(ValueError):
            HideSeekV7Env(layouts=(LAYOUT,), spawns_by_layout=())


if __name__ == "__main__":
    unittest.main()
