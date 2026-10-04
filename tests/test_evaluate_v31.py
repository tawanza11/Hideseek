"""Legal-action selection uses the public V3 observation only."""

import unittest

import numpy as np

from env.hide_seek_v3_env import DEFAULT_MAP, HideSeekV3Env
from evaluation.evaluate_v31 import legal_actions_from_observation


class LegalActionsTests(unittest.TestCase):
    def test_walls_and_boundaries_are_excluded(self) -> None:
        env = HideSeekV3Env(map_rows=("...", ".#.", "..."))
        env.reset(seed=1)
        env.seeker_pos, env.hider_pos = (0, 1), (2, 1)

        actions = legal_actions_from_observation(env._observation())

        self.assertEqual(actions, (0, 1))
        self.assertFalse(env.is_visible("seeker"))

    def test_visibility_does_not_change_legal_actions(self) -> None:
        env = HideSeekV3Env(map_rows=("...", ".#.", "..."))
        env.reset(seed=2)
        env.seeker_pos, env.hider_pos = (0, 0), (2, 2)
        hidden_actions = legal_actions_from_observation(env._observation())
        env.hider_pos = (0, 2)
        visible_actions = legal_actions_from_observation(env._observation())

        self.assertEqual(hidden_actions, (1, 3))
        self.assertEqual(visible_actions, hidden_actions)

    def test_invalid_observation_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            legal_actions_from_observation(np.zeros(10, dtype=np.float32))

    def test_every_default_map_cell_matches_environment_movement(self) -> None:
        env = HideSeekV3Env()
        env.reset(seed=3)
        for y, row in enumerate(DEFAULT_MAP):
            for x, cell in enumerate(row):
                if cell == "#":
                    continue
                env.seeker_pos = (x, y)
                expected = tuple(
                    index
                    for index, (dx, dy) in enumerate(env._ACTION_DELTAS)
                    if env._inside((x + dx, y + dy))
                )
                with self.subTest(position=(x, y)):
                    self.assertEqual(legal_actions_from_observation(env._observation()), expected)


if __name__ == "__main__":
    unittest.main()
