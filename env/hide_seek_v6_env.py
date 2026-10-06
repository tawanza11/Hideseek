"""V4 Seeker rules with one frozen opponent selected per episode."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from env.hide_seek_v4_env import HideSeekV4Env


class HideSeekV6Env(HideSeekV4Env):
    """Rotate opponents at reset while retaining V4's no-memory game rules."""

    def __init__(
        self,
        *,
        map_pool: tuple[tuple[str, ...], ...],
        opponent_policies: tuple[Callable[[np.ndarray], int] | None, ...],
        novelty_bonus: float = 0.01,
        max_steps: int = 100,
    ) -> None:
        if not opponent_policies:
            raise ValueError("opponent_policies must contain at least one policy")
        if any(policy is not None and not callable(policy) for policy in opponent_policies):
            raise TypeError("opponent_policies entries must be callable or None")
        self.opponent_policies = opponent_policies
        self.opponent_index = 0
        super().__init__(
            map_pool=map_pool,
            visit_memory=False,
            novelty_bonus=novelty_bonus,
            max_steps=max_steps,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, bool | int]]:
        observation, info = super().reset(seed=seed, options=options)
        self.opponent_index = (
            int(self.np_random.integers(len(self.opponent_policies)))
            if len(self.opponent_policies) > 1
            else 0
        )
        self.opponent_policy = self.opponent_policies[self.opponent_index]
        info["opponent_index"] = self.opponent_index
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
        observation, reward, terminated, truncated, info = super().step(action)
        info["opponent_index"] = self.opponent_index
        return observation, reward, terminated, truncated, info
