"""Focused standard-library tests for the Hide-and-Seek environment."""

import unittest

import numpy as np

from env.hide_seek_env import HideSeekEnv


class HideSeekEnvTests(unittest.TestCase):
    def test_reset_is_seeded_and_observation_matches_space(self) -> None:
        first = HideSeekEnv(grid_size=5)
        second = HideSeekEnv(grid_size=5)
        first_observation, first_info = first.reset(seed=123)
        second_observation, second_info = second.reset(seed=123)

        np.testing.assert_array_equal(first_observation, second_observation)
        self.assertEqual(first.seeker_pos, second.seeker_pos)
        self.assertEqual(first.hider_pos, second.hider_pos)
        self.assertNotEqual(first.seeker_pos, first.hider_pos)
        self.assertEqual(first_observation.dtype, np.float32)
        self.assertTrue(first.observation_space.contains(first_observation))
        self.assertEqual(first_info["distance"], second_info["distance"])

    def test_capture_terminates_with_positive_reward(self) -> None:
        env = HideSeekEnv(grid_size=4)
        env.reset(seed=1)
        env.seeker_pos = (0, 0)
        env.hider_pos = (1, 0)

        observation, reward, terminated, truncated, info = env.step(3)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertTrue(info["captured"])
        self.assertEqual(info["distance"], 0)

    def test_timeout_truncates_with_negative_reward(self) -> None:
        env = HideSeekEnv(grid_size=4, max_steps=1)
        env.reset(seed=2)
        env.seeker_pos = (0, 0)
        env.hider_pos = (3, 3)

        _, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(reward, -1.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info["success"])
        self.assertFalse(info["captured"])

    def test_rgb_render_has_expected_dimensions(self) -> None:
        env = HideSeekEnv(grid_size=3, render_mode="rgb_array")
        env.reset(seed=3)

        image = env.render()

        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, (144, 144, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_rejects_invalid_configuration_and_action(self) -> None:
        with self.assertRaises(ValueError):
            HideSeekEnv(grid_size=1)
        with self.assertRaises(ValueError):
            HideSeekEnv(max_steps=0)
        env = HideSeekEnv()
        env.reset(seed=4)
        with self.assertRaises(ValueError):
            env.step(4)

    def test_cannot_step_after_episode_ends(self) -> None:
        env = HideSeekEnv(grid_size=4, max_steps=1)
        env.reset(seed=5)
        env.seeker_pos = (0, 0)
        env.hider_pos = (3, 3)
        env.step(0)
        with self.assertRaises(RuntimeError):
            env.step(0)


if __name__ == "__main__":
    unittest.main()
