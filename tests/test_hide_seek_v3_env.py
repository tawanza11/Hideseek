"""Focused tests for the V3 walls, vision, and observation contract."""

import unittest

import numpy as np

from env.hide_seek_v3_env import DEFAULT_MAP, HideSeekV3Env


class HideSeekV3EnvTests(unittest.TestCase):
    def test_default_map_is_valid_and_reset_spawns_on_free_cells(self) -> None:
        env = HideSeekV3Env()
        observation, info = env.reset(seed=13)

        self.assertEqual(len(DEFAULT_MAP), 10)
        self.assertTrue(all(len(row) == 10 for row in DEFAULT_MAP))
        self.assertTrue(env._inside(env.seeker_pos))
        self.assertTrue(env._inside(env.hider_pos))
        self.assertNotEqual(env.seeker_pos, env.hider_pos)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(observation.shape, (105,))
        self.assertIn(info["distance"], (-1, env._distance()))

    def test_rejects_invalid_maps(self) -> None:
        invalid_maps = (
            (),
            ("..",),
            ("##", "##"),
            ("..", "x."),
            (".#.", "###", ".#."),
        )
        for map_rows in invalid_maps:
            with self.subTest(map_rows=map_rows):
                with self.assertRaises(ValueError):
                    HideSeekV3Env(map_rows=map_rows)

    def test_wall_blocks_movement_and_spawns_avoid_walls(self) -> None:
        env = HideSeekV3Env(map_rows=("...", ".#.", "..."))
        env.reset(seed=2)
        env.seeker_pos = (0, 1)
        env.hider_pos = (2, 2)

        env.step(3)

        self.assertEqual(env.seeker_pos, (0, 1))
        env.reset(seed=9)
        self.assertTrue(env._inside(env.seeker_pos))
        self.assertTrue(env._inside(env.hider_pos))

    def test_line_of_sight_open_blocked_and_diagonal_corner(self) -> None:
        open_env = HideSeekV3Env(map_rows=("...", "...", "..."))
        open_env.seeker_pos, open_env.hider_pos = (0, 0), (2, 2)
        self.assertTrue(open_env.is_visible("seeker"))

        blocked_env = HideSeekV3Env(map_rows=("...", ".#.", "..."))
        blocked_env.seeker_pos, blocked_env.hider_pos = (0, 1), (2, 1)
        self.assertFalse(blocked_env.is_visible("seeker"))

        corner_env = HideSeekV3Env(map_rows=(".#.", "...", "..."))
        corner_env.seeker_pos, corner_env.hider_pos = (0, 0), (1, 1)
        self.assertFalse(corner_env.is_visible("seeker"))

    def test_hidden_opponent_is_masked_and_distance_is_not_reported(self) -> None:
        env = HideSeekV3Env(
            map_rows=("...", ".#.", "..."),
            opponent_policy=lambda _: 1,
        )
        env.reset(seed=3)
        env.seeker_pos, env.hider_pos = (0, 1), (2, 1)

        observation = env._observation()
        self.assertFalse(env.is_visible())
        np.testing.assert_array_equal(observation[2:5], [0, 0, 0])
        self.assertEqual(env._reported_distance(), -1)
        _, _, _, _, info = env.step(0)
        self.assertEqual(info["distance"], -1)

    def test_visible_opponent_is_encoded_with_wall_map(self) -> None:
        env = HideSeekV3Env(map_rows=("...", "...", "..."))
        env.reset(seed=5)
        env.seeker_pos, env.hider_pos = (0, 0), (2, 2)

        observation = env._observation()

        np.testing.assert_allclose(observation[:5], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(observation[5:], np.zeros(9, dtype=np.float32))
        self.assertEqual(env._reported_distance(), 4)

    def test_rgb_render_paints_wall_cells(self) -> None:
        env = HideSeekV3Env(map_rows=("...", ".#.", "..."), render_mode="rgb_array")
        env.reset(seed=6)
        env.seeker_pos, env.hider_pos = (0, 0), (2, 2)

        image = env.render()

        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, (144, 144, 3))
        np.testing.assert_array_equal(image[72, 72], [45, 45, 45])


if __name__ == "__main__":
    unittest.main()
