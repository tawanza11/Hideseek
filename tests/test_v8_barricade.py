"""The V8 demonstration reads only visible state and closes each doorway."""

import unittest

from env.hide_seek_v7_env import HideSeekV7Env
from env.v8_maps import TRAIN_LAYOUTS
from training.v8_barricade import barricade_direction


class BarricadeTests(unittest.TestCase):
    def test_every_training_orientation_has_a_visible_legal_push(self) -> None:
        for name, challenge in TRAIN_LAYOUTS.items():
            with self.subTest(map=name):
                env = HideSeekV7Env(layouts=(challenge.arena,), role="hider")
                observation, _ = env.reset(seed=1, options={
                    "seeker_pos": challenge.seeker_spawn,
                    "hider_pos": challenge.hider_spawn,
                })
                action = barricade_direction(observation)
                self.assertIsNotNone(action)
                env.step(action)
                dx = challenge.arena.block[0] - challenge.hider_spawn[0]
                dy = challenge.arena.block[1] - challenge.hider_spawn[1]
                target = (challenge.arena.block[0] + dx, challenge.arena.block[1] + dy)
                self.assertEqual(env.blocks, {target})
                self.assertIn(("hider", "block_pushed"), env.last_events)
                env.close()


if __name__ == "__main__":
    unittest.main()
