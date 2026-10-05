"""V4 map splits stay valid, distinct, and compatible with V3 models."""

import unittest

from env.hide_seek_v3_env import DEFAULT_MAP, HideSeekV3Env
from env.v4_maps import TEST_MAPS, TRAIN_MAPS, VALIDATION_MAPS


class V4MapTests(unittest.TestCase):
    def test_partitions_are_distinct_connected_10x10_maps(self) -> None:
        maps = (DEFAULT_MAP, *TRAIN_MAPS.values(), *VALIDATION_MAPS.values(), *TEST_MAPS.values())
        self.assertEqual(len(maps), len(set(maps)))
        for rows in maps:
            with self.subTest(map_rows=rows):
                env = HideSeekV3Env(map_rows=rows)
                observation, _ = env.reset(seed=7)
                self.assertEqual(observation.shape, (105,))
                self.assertTrue(env._inside(env.seeker_pos))
                self.assertTrue(env._inside(env.hider_pos))


if __name__ == "__main__":
    unittest.main()
