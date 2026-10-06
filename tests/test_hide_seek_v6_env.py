"""V6 opponent rotation and V4 rule compatibility."""

import unittest

import numpy as np

from env.hide_seek_v3_env import DEFAULT_MAP
from env.hide_seek_v4_env import HideSeekV4Env
from env.hide_seek_v6_env import HideSeekV6Env
from env.v4_maps import TRAIN_MAPS


class HideSeekV6EnvTests(unittest.TestCase):
    def test_opponent_selection_is_deterministic_for_seed(self) -> None:
        policies = (None, lambda _: 0, lambda _: 3)
        first = HideSeekV6Env(map_pool=(DEFAULT_MAP,), opponent_policies=policies)
        second = HideSeekV6Env(map_pool=(DEFAULT_MAP,), opponent_policies=policies)
        selected = set()
        for seed in range(30):
            first_observation, first_info = first.reset(seed=seed)
            second_observation, second_info = second.reset(seed=seed)
            np.testing.assert_array_equal(first_observation, second_observation)
            self.assertEqual(first_info, second_info)
            self.assertIs(first.opponent_policy, policies[first_info["opponent_index"]])
            selected.add(first_info["opponent_index"])
        self.assertEqual(selected, {0, 1, 2})

    def test_opponent_stays_fixed_within_episode(self) -> None:
        calls = [0, 0]

        def first_policy(_: np.ndarray) -> int:
            calls[0] += 1
            return 0

        def second_policy(_: np.ndarray) -> int:
            calls[1] += 1
            return 3

        env = HideSeekV6Env(
            map_pool=(DEFAULT_MAP,),
            opponent_policies=(first_policy, second_policy),
            max_steps=10,
        )
        _, reset_info = env.reset(seed=7)
        selected = reset_info["opponent_index"]
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(0)
            self.assertEqual(info["opponent_index"], selected)
            self.assertIs(env.opponent_policy, env.opponent_policies[selected])
            if terminated or truncated:
                break
        self.assertGreater(calls[selected], 0)
        self.assertEqual(calls[1 - selected], 0)

    def test_single_opponent_matches_v4_rules(self) -> None:
        policy = lambda _: 3
        old = HideSeekV4Env(
            map_pool=(DEFAULT_MAP,), visit_memory=False, opponent_policy=policy
        )
        new = HideSeekV6Env(map_pool=(DEFAULT_MAP,), opponent_policies=(policy,))
        old_observation, old_info = old.reset(seed=17)
        new_observation, new_info = new.reset(seed=17)
        np.testing.assert_array_equal(old_observation, new_observation)
        self.assertEqual(old_info, {key: value for key, value in new_info.items() if key != "opponent_index"})
        for action in (0, 1, 3, 2, 0, 3, 1):
            old_step = old.step(action)
            new_step = new.step(action)
            np.testing.assert_array_equal(old_step[0], new_step[0])
            self.assertEqual(old_step[1:4], new_step[1:4])
            self.assertEqual(
                old_step[4],
                {key: value for key, value in new_step[4].items() if key != "opponent_index"},
            )
            if old_step[2] or old_step[3]:
                break

    def test_map_pool_can_change_between_episodes(self) -> None:
        env = HideSeekV6Env(
            map_pool=(DEFAULT_MAP,), opponent_policies=(None,)
        )
        env.reset(seed=1)
        env.map_pool = (TRAIN_MAPS["T1"],)
        observation, info = env.reset(seed=1)
        self.assertEqual(env.map_rows, TRAIN_MAPS["T1"])
        self.assertEqual(observation.shape, (105,))
        self.assertEqual(info["map_index"], 0)

    def test_rejects_empty_or_invalid_opponent_pool(self) -> None:
        with self.assertRaises(ValueError):
            HideSeekV6Env(map_pool=(DEFAULT_MAP,), opponent_policies=())
        with self.assertRaises(TypeError):
            HideSeekV6Env(map_pool=(DEFAULT_MAP,), opponent_policies=(1,))


if __name__ == "__main__":
    unittest.main()
