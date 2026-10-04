"""V3 seeker training environment with a penalty for blocked moves."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import numpy as np

from env.hide_seek_v3_env import HideSeekV3Env


class SeekerWallPenaltyEnv(HideSeekV3Env):
    """Keep V3 observations and rules, but discourage walking into walls."""

    def __init__(
        self,
        *,
        wall_penalty: float = 0.02,
        map_rows: tuple[str, ...] | None = None,
        max_steps: int = 100,
        opponent_policy: Callable[[np.ndarray], int] | None = None,
    ) -> None:
        if not isfinite(wall_penalty) or wall_penalty <= 0:
            raise ValueError("wall_penalty must be finite and positive")
        self.wall_penalty = float(wall_penalty)
        super().__init__(
            map_rows=map_rows,
            max_steps=max_steps,
            role="seeker",
            opponent_policy=opponent_policy,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
        previous_pos = self.seeker_pos
        observation, reward, terminated, truncated, info = super().step(action)
        if not terminated and self.seeker_pos == previous_pos:
            reward -= self.wall_penalty
        return observation, reward, terminated, truncated, info
