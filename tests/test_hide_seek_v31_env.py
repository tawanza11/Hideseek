"""V3.1 wall penalty changes training rewards without changing V3 rules."""

import math
import unittest

from env.hide_seek_v3_env import HideSeekV3Env
from env.hide_seek_v31_env import SeekerWallPenaltyEnv


class SeekerWallPenaltyEnvTests(unittest.TestCase):
    def test_only_blocked_move_is_penalized(self) -> None:
        env = SeekerWallPenaltyEnv(
            map_rows=("...", ".#.", "..."),
            opponent_policy=lambda _: 3,
        )
        env.reset(seed=1)
        env.seeker_pos = (0, 1)
        env.hider_pos = (2, 2)

        _, blocked_reward, terminated, truncated, _ = env.step(3)
        self.assertEqual(env.seeker_pos, (0, 1))
        self.assertAlmostEqual(blocked_reward, -0.03)
        self.assertFalse(terminated or truncated)

        _, legal_reward, _, _, _ = env.step(0)
        self.assertEqual(env.seeker_pos, (0, 0))
        self.assertAlmostEqual(legal_reward, -0.01)

    def test_capture_reward_and_original_v3_remain_unchanged(self) -> None:
        shaped = SeekerWallPenaltyEnv(map_rows=("...", "...", "..."))
        shaped.reset(seed=2)
        shaped.seeker_pos, shaped.hider_pos = (0, 0), (0, 1)
        _, reward, terminated, _, _ = shaped.step(1)
        self.assertTrue(terminated)
        self.assertEqual(reward, 1.0)

        original = HideSeekV3Env(
            map_rows=("...", ".#.", "..."),
            opponent_policy=lambda _: 3,
        )
        original.reset(seed=3)
        original.seeker_pos, original.hider_pos = (0, 1), (2, 2)
        _, original_reward, _, _, _ = original.step(3)
        self.assertAlmostEqual(original_reward, -0.01)

    def test_penalty_must_be_finite_and_positive(self) -> None:
        for penalty in (0.0, -0.1, math.inf, math.nan):
            with self.subTest(penalty=penalty):
                with self.assertRaises(ValueError):
                    SeekerWallPenaltyEnv(wall_penalty=penalty)


if __name__ == "__main__":
    unittest.main()
