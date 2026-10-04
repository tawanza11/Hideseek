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

    def test_default_seeker_behavior_remains_seeded(self) -> None:
        first = HideSeekEnv(grid_size=5, max_steps=8)
        second = HideSeekEnv(grid_size=5, max_steps=8, role="seeker")
        np.testing.assert_array_equal(first.reset(seed=21)[0], second.reset(seed=21)[0])
        for action in (0, 3, 1, 2):
            first_result = first.step(action)
            second_result = second.step(action)
            np.testing.assert_array_equal(first_result[0], second_result[0])
            self.assertEqual(first_result[1:], second_result[1:])
            if first_result[2] or first_result[3]:
                break

    def test_seeker_capture_uses_callback_with_hider_centric_observation(self) -> None:
        callback_observations: list[np.ndarray] = []

        def keep_hider_still(observation: np.ndarray) -> int:
            callback_observations.append(observation.copy())
            return 2

        env = HideSeekEnv(grid_size=4, opponent_policy=keep_hider_still)
        env.reset(seed=1)
        env.seeker_pos = (0, 0)
        env.hider_pos = (2, 0)

        _, reward, terminated, truncated, info = env.step(3)

        np.testing.assert_allclose(callback_observations[0], [2 / 3, 0, 0, 0, 2 / 3, 1 / 3, 0, 1])
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertTrue(info["captured"])

    def test_hider_role_moves_after_seeker_and_observes_seeker_centric_state(self) -> None:
        callback_observations: list[np.ndarray] = []

        def move_seeker_right(observation: np.ndarray) -> int:
            callback_observations.append(observation.copy())
            return 3

        env = HideSeekEnv(grid_size=4, role="hider", opponent_policy=move_seeker_right)
        env.reset(seed=2)
        env.seeker_pos = (0, 0)
        env.hider_pos = (3, 3)

        observation, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(callback_observations[0].tolist(), [0, 0, 1, 1, 0, 1, 0, 1])
        self.assertEqual(env.seeker_pos, (1, 0))
        self.assertEqual(env.hider_pos, (3, 2))
        np.testing.assert_allclose(observation[:4], [1, 2 / 3, 1 / 3, 0])
        self.assertEqual(reward, 0.01)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["success"])
        self.assertFalse(info["captured"])

    def test_hider_can_be_captured_before_its_move(self) -> None:
        env = HideSeekEnv(grid_size=4, role="hider", opponent_policy=lambda _: 3)
        env.reset(seed=3)
        env.seeker_pos = (0, 0)
        env.hider_pos = (1, 0)

        _, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(env.hider_pos, (1, 0))
        self.assertEqual(reward, -1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["success"])
        self.assertTrue(info["captured"])

    def test_hider_survival_timeout_is_success(self) -> None:
        env = HideSeekEnv(grid_size=4, max_steps=1, role="hider", opponent_policy=lambda _: 0)
        env.reset(seed=4)
        env.seeker_pos = (0, 0)
        env.hider_pos = (3, 3)

        _, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(reward, 1.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["success"])
        self.assertFalse(info["captured"])

    def test_rejects_invalid_opponent_action(self) -> None:
        env = HideSeekEnv(opponent_policy=lambda _: 9)
        env.reset(seed=5)
        env.seeker_pos = (0, 0)
        env.hider_pos = (3, 3)
        with self.assertRaises(ValueError):
            env.step(0)

    def test_random_seeker_is_seeded_in_hider_role(self) -> None:
        first = HideSeekEnv(grid_size=5, role="hider")
        second = HideSeekEnv(grid_size=5, role="hider")
        np.testing.assert_array_equal(first.reset(seed=37)[0], second.reset(seed=37)[0])
        for action in (0, 3, 1, 2):
            first_result = first.step(action)
            second_result = second.step(action)
            np.testing.assert_array_equal(first_result[0], second_result[0])
            self.assertEqual(first_result[1:], second_result[1:])
            if first_result[2] or first_result[3]:
                break

    def test_both_roles_resolve_the_same_turn(self) -> None:
        seeker_env = HideSeekEnv(grid_size=4, opponent_policy=lambda _: 2)
        hider_env = HideSeekEnv(grid_size=4, role="hider", opponent_policy=lambda _: 3)
        seeker_env.reset(seed=11)
        hider_env.reset(seed=11)
        seeker_env.seeker_pos = hider_env.seeker_pos = (0, 0)
        seeker_env.hider_pos = hider_env.hider_pos = (2, 0)

        seeker_env.step(3)
        hider_env.step(2)

        self.assertEqual(seeker_env.seeker_pos, hider_env.seeker_pos)
        self.assertEqual(seeker_env.hider_pos, hider_env.hider_pos)
        self.assertEqual(seeker_env.steps, hider_env.steps)


if __name__ == "__main__":
    unittest.main()
