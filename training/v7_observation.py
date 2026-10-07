"""Observable route hints for a learned V7 Seeker policy."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from training.v7_teacher import search_features


def with_route_hints(observation: np.ndarray) -> np.ndarray:
    if observation.shape != (508,):
        raise ValueError("Expected the 508-value V7 observation")
    return np.concatenate((observation, search_features(observation)))


class SeekerPlannerObservation(gym.ObservationWrapper):
    """Append only route hints computable from the Seeker's own observation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if env.observation_space.shape != (508,):
            raise ValueError("SeekerPlannerObservation needs a V7 environment")
        self.observation_space = spaces.Box(
            low=np.zeros(516, dtype=np.float32),
            high=np.ones(516, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return with_route_hints(observation)
