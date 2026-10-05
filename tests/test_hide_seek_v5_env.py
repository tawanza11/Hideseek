"""Legal masks use only the Seeker's visible state."""

import unittest

import numpy as np

from env.hide_seek_v3_env import DEFAULT_MAP
from env.hide_seek_v4_env import HideSeekV4Env
from env.hide_seek_v5_env import HideSeekV5Env


class HideSeekV5EnvTests(unittest.TestCase):
    def test_masks_match_moves_from_observation(self) -> None:
        env = HideSeekV5Env(map_pool=(DEFAULT_MAP,))
        for seed in range(20):
            observation, _ = env.reset(seed=seed)
            mask = env.action_masks()
            self.assertEqual(mask.shape, (4,))
            self.assertEqual(mask.dtype, np.dtype(bool))
            self.assertTrue(mask.any())
            x = round(float(observation[0]) * 9)
            y = round(float(observation[1]) * 9)
            for action, (dx, dy) in enumerate(env._ACTION_DELTAS):
                target_x, target_y = x + dx, y + dy
                expected = (
                    0 <= target_x < 10
                    and 0 <= target_y < 10
                    and observation[5 + target_y * 10 + target_x] == 0
                )
                self.assertEqual(bool(mask[action]), expected)

    def test_hidden_hider_position_does_not_change_mask(self) -> None:
        env = HideSeekV5Env(map_pool=(DEFAULT_MAP,))
        env.reset(seed=7)
        env.seeker_pos = (1, 1)
        env.hider_pos = (6, 1)
        self.assertFalse(env.is_visible("seeker"))
        first = env.action_masks()
        env.hider_pos = (7, 7)
        self.assertFalse(env.is_visible("seeker"))
        np.testing.assert_array_equal(first, env.action_masks())

    def test_masked_environment_keeps_control_game_rules(self) -> None:
        control = HideSeekV4Env(map_pool=(DEFAULT_MAP,), visit_memory=False)
        masked = HideSeekV5Env(map_pool=(DEFAULT_MAP,))
        control_observation, _ = control.reset(seed=17)
        masked_observation, _ = masked.reset(seed=17)
        np.testing.assert_array_equal(control_observation, masked_observation)
        for action in (0, 1, 3, 2, 0, 3, 1):
            control_step = control.step(action)
            masked_step = masked.step(action)
            np.testing.assert_array_equal(control_step[0], masked_step[0])
            self.assertEqual(control_step[1:], masked_step[1:])
            if control_step[2] or control_step[3]:
                break


if __name__ == "__main__":
    unittest.main()
