"""Seeker environment that rotates training maps and tracks visited cells."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.hide_seek_v3_env import HideSeekV3Env


class HideSeekV4Env(HideSeekV3Env):
    """V3 rules with a fixed-size optional visit map in the Seeker observation."""

    def __init__(
        self,
        *,
        map_pool: tuple[tuple[str, ...], ...],
        visit_memory: bool,
        novelty_bonus: float = 0.01,
        max_steps: int = 100,
        opponent_policy: Callable[[np.ndarray], int] | None = None,
    ) -> None:
        if not map_pool:
            raise ValueError("map_pool must contain at least one map")
        grid_size = HideSeekV3Env._validate_map(map_pool[0])
        for rows in map_pool[1:]:
            if HideSeekV3Env._validate_map(rows) != grid_size:
                raise ValueError("all maps must have the same grid size")
        if not isfinite(novelty_bonus) or novelty_bonus < 0:
            raise ValueError("novelty_bonus must be finite and nonnegative")
        self.map_pool = map_pool
        self.visit_memory = visit_memory
        self.novelty_bonus = float(novelty_bonus)
        self.map_index = 0
        self._visited = np.zeros(grid_size * grid_size, dtype=np.float32)
        super().__init__(
            map_rows=map_pool[0],
            max_steps=max_steps,
            role="seeker",
            opponent_policy=opponent_policy,
        )
        if visit_memory:
            observation_size = 5 + 2 * grid_size * grid_size
            self.observation_space = spaces.Box(
                low=np.zeros(observation_size, dtype=np.float32),
                high=np.ones(observation_size, dtype=np.float32),
                dtype=np.float32,
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, bool | int]]:
        gym.Env.reset(self, seed=seed)
        self.map_index = (
            int(self.np_random.integers(len(self.map_pool)))
            if len(self.map_pool) > 1
            else 0
        )
        self.map_rows = self.map_pool[self.map_index]
        self._free_cells = tuple(
            y * self.grid_size + x
            for y, row in enumerate(self.map_rows)
            for x, cell in enumerate(row)
            if cell == "."
        )
        self._visited.fill(0)
        _, info = super().reset(seed=None, options=options)
        self._mark_visited()
        info["map_index"] = self.map_index
        return self._observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
        visible_before = self.is_visible("seeker")
        _, reward, terminated, truncated, info = super().step(action)
        new_cell = self._mark_visited()
        if new_cell and not visible_before and not terminated and not truncated:
            reward += self.novelty_bonus
        info["map_index"] = self.map_index
        return self._observation(), reward, terminated, truncated, info

    def _observation_for(self, role: Literal["seeker", "hider"]) -> np.ndarray:
        base = super()._observation_for(role)
        if role == "seeker" and self.visit_memory:
            return np.concatenate((base, self._visited))
        return base

    def _mark_visited(self) -> bool:
        x, y = self.seeker_pos
        index = y * self.grid_size + x
        new_cell = self._visited[index] == 0
        self._visited[index] = 1
        return bool(new_cell)
