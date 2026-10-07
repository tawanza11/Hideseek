"""Observable route hints must not depend on an unseen opponent."""

import unittest

import numpy as np

from env.hide_seek_v7_env import HideSeekV7Env, ObjectLayout
from training.v7_teacher import legal_actions, search_action, search_features


LAYOUT = ObjectLayout(
    ("#####", "#...#", "#.#.#", "#...#", "#####"),
    block=(2, 1), ramp=(1, 2), low_wall=(2, 2),
)


class V7TeacherTests(unittest.TestCase):
    def test_hidden_opponent_position_does_not_change_seeker_hints(self) -> None:
        env = HideSeekV7Env(layouts=(LAYOUT,))
        first, _ = env.reset(
            seed=1, options={"seeker_pos": (1, 2), "hider_pos": (3, 2)}
        )
        second, _ = env.reset(
            seed=1, options={"seeker_pos": (1, 2), "hider_pos": (3, 1)}
        )
        self.assertEqual(first[4], 0)
        self.assertEqual(second[4], 0)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(search_features(first), search_features(second))
        self.assertEqual(search_action(first), search_action(second))

    def test_actions_include_push_and_ramp_only_when_physically_possible(self) -> None:
        env = HideSeekV7Env(layouts=(LAYOUT,))
        observation, _ = env.reset(
            seed=2, options={"seeker_pos": (1, 2), "hider_pos": (3, 3)}
        )
        self.assertIn(3, legal_actions(observation))  # Ramp crosses the low wall.
        env.seeker_pos = (1, 1)
        observation = env._observation_for("seeker")
        self.assertIn(3, legal_actions(observation))  # Block can be pushed right.
        self.assertEqual(search_features(observation).shape, (8,))


if __name__ == "__main__":
    unittest.main()
