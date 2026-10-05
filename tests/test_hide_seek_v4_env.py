"""V4 map rotation, visitation memory, and reward boundaries."""

import unittest

import numpy as np

from env.hide_seek_v3_env import HideSeekV3Env
from env.hide_seek_v4_env import HideSeekV4Env
from env.v4_maps import TRAIN_MAPS, VALIDATION_MAPS


class HideSeekV4EnvTests(unittest.TestCase):
    def test_map_rotation_and_observation_contracts(self) -> None:
        pool = tuple(TRAIN_MAPS.values())
        env = HideSeekV4Env(map_pool=pool, visit_memory=True)
        selected = set()
        for seed in range(20):
            observation, info = env.reset(seed=seed)
            selected.add(info["map_index"])
            self.assertEqual(env.map_rows, pool[info["map_index"]])
            self.assertEqual(observation.shape, (205,))
            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(int(np.sum(observation[105:])), 1)
            self.assertEqual(env._observation_for("hider").shape, (105,))
        self.assertEqual(selected, {0, 1})

    def test_new_hidden_cell_gets_bonus_once(self) -> None:
        env = HideSeekV4Env(
            map_pool=(("...", ".#.", "..."),),
            visit_memory=True,
            novelty_bonus=0.01,
            opponent_policy=lambda _: 3,
        )
        env.reset(seed=1)
        env.seeker_pos, env.hider_pos = (0, 1), (2, 1)
        env._visited.fill(0)
        env._mark_visited()

        observation, reward, _, _, _ = env.step(0)
        self.assertEqual(env.seeker_pos, (0, 0))
        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(int(np.sum(observation[14:])), 2)

        _, revisit_reward, _, _, _ = env.step(1)
        self.assertEqual(env.seeker_pos, (0, 1))
        self.assertAlmostEqual(revisit_reward, -0.01)

    def test_no_memory_variant_keeps_v3_observation_size(self) -> None:
        env = HideSeekV4Env(map_pool=(TRAIN_MAPS["T1"],), visit_memory=False)
        observation, _ = env.reset(seed=5)
        self.assertEqual(observation.shape, (105,))

    def test_single_map_uses_same_spawns_as_v3(self) -> None:
        rows = VALIDATION_MAPS["E1"]
        old = HideSeekV3Env(map_rows=rows)
        new = HideSeekV4Env(map_pool=(rows,), visit_memory=True)
        for seed in (100_000, 100_001, 100_050):
            old.reset(seed=seed)
            new.reset(seed=seed)
            self.assertEqual((new.seeker_pos, new.hider_pos), (old.seeker_pos, old.hider_pos))

    def test_failed_last_step_keeps_full_failure_reward(self) -> None:
        env = HideSeekV4Env(
            map_pool=(("...", ".#.", "..."),),
            visit_memory=True,
            novelty_bonus=0.01,
            max_steps=1,
            opponent_policy=lambda _: 3,
        )
        env.reset(seed=1)
        env.seeker_pos, env.hider_pos = (0, 1), (2, 1)
        env._visited.fill(0)
        env._mark_visited()
        _, reward, terminated, truncated, _ = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, -1.0)

    def test_rejects_invalid_pool(self) -> None:
        with self.assertRaises(ValueError):
            HideSeekV4Env(map_pool=(), visit_memory=True)
        with self.assertRaises(ValueError):
            HideSeekV4Env(
                map_pool=(("..", ".."), ("...", "...", "...")),
                visit_memory=True,
            )


if __name__ == "__main__":
    unittest.main()
