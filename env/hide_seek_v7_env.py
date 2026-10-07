"""Object-aware grid game. The 3D viewer only visualizes these grid rules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.hide_seek_env import HideSeekEnv
from env.hide_seek_v3_env import HideSeekV3Env


Cell = tuple[int, int]


@dataclass(frozen=True)
class ObjectLayout:
    map_rows: tuple[str, ...]
    block: Cell
    ramp: Cell
    low_wall: Cell

    def validate(self) -> int:
        size = HideSeekV3Env._validate_map(self.map_rows)
        if len({self.block, self.ramp, self.low_wall}) != 3:
            raise ValueError("block, ramp and low wall must be distinct")
        for name, cell in (("block", self.block), ("ramp", self.ramp)):
            x, y = cell
            if not (0 <= x < size and 0 <= y < size) or self.map_rows[y][x] != ".":
                raise ValueError(f"{name} must start on a floor cell")
        x, y = self.low_wall
        if not (0 <= x < size and 0 <= y < size) or self.map_rows[y][x] != "#":
            raise ValueError("low wall must replace an existing wall cell")
        dx, dy = x - self.ramp[0], y - self.ramp[1]
        landing = (x + dx, y + dy)
        if abs(dx) + abs(dy) != 1 or not (
            0 <= landing[0] < size and 0 <= landing[1] < size
        ) or self.map_rows[landing[1]][landing[0]] != ".":
            raise ValueError("ramp must face the low wall with floor beyond")
        if self.block == landing:
            raise ValueError("block cannot occupy the ramp landing")
        def floor(cell: Cell) -> bool:
            return (0 <= cell[0] < size and 0 <= cell[1] < size
                    and self.map_rows[cell[1]][cell[0]] == ".")
        if not any(
            floor((self.block[0] - dx, self.block[1] - dy))
            and floor((self.block[0] + dx, self.block[1] + dy))
            for dx, dy in ((1, 0), (0, 1))
        ):
            raise ValueError("block must be pushable from a floor cell")
        if sum(cell == "." for row in self.map_rows for cell in row) < 4:
            raise ValueError("layout needs free spawn cells")
        return size


class HideSeekV7Env(HideSeekV3Env):
    """Four moves, automatic block pushing, and ramp-assisted wall crossing.

    Both roles receive only their own visible opponent position. Static walls,
    blocks, ramps, low walls, own visits, and own last sighting are observable.
    """

    def __init__(
        self,
        *,
        layouts: tuple[ObjectLayout, ...],
        role: Literal["seeker", "hider"] = "seeker",
        max_steps: int = 100,
        opponent_policy: Callable[[np.ndarray], int] | None = None,
        opponent_policies: tuple[Callable[[np.ndarray], int] | None, ...] | None = None,
        novelty_bonus: float = 0.0,
        enable_blocks: bool = True,
        enable_ramp: bool = True,
    ) -> None:
        if not layouts:
            raise ValueError("layouts must not be empty")
        size = layouts[0].validate()
        if any(layout.validate() != size for layout in layouts[1:]):
            raise ValueError("all layouts must have one grid size")
        if not 0 <= novelty_bonus < 1:
            raise ValueError("novelty_bonus must be in [0, 1)")
        if opponent_policies is not None and (not opponent_policies or opponent_policy is not None):
            raise ValueError("provide one nonempty opponent pool or one opponent policy")
        if opponent_policies is not None and any(
            policy is not None and not callable(policy) for policy in opponent_policies
        ):
            raise TypeError("opponent policies must be callable or None")
        self.opponent_policies = opponent_policies
        self.opponent_index = 0
        self.layouts = layouts
        self.layout_index = 0
        self.blocks: set[Cell] = {layouts[0].block} if enable_blocks else set()
        self.ramp = layouts[0].ramp
        self.low_wall = layouts[0].low_wall
        self.enable_blocks = enable_blocks
        self.enable_ramp = enable_ramp
        self.novelty_bonus = novelty_bonus
        self._visits = {
            "seeker": np.zeros(size * size, dtype=np.float32),
            "hider": np.zeros(size * size, dtype=np.float32),
        }
        self._last_seen: dict[str, Cell | None] = {"seeker": None, "hider": None}
        self.last_events: list[tuple[str, str]] = []
        super().__init__(
            map_rows=layouts[0].map_rows,
            role=role,
            max_steps=max_steps,
            opponent_policy=opponent_policy,
        )
        length = 8 + 5 * size * size
        self.observation_space = spaces.Box(
            low=np.zeros(length, dtype=np.float32),
            high=np.ones(length, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, bool | int]]:
        gym.Env.reset(self, seed=seed)
        self.layout_index = int(self.np_random.integers(len(self.layouts)))
        layout = self.layouts[self.layout_index]
        self.map_rows = layout.map_rows
        self._free_cells = tuple(
            y * self.grid_size + x
            for y, row in enumerate(self.map_rows)
            for x, cell in enumerate(row)
            if cell == "."
        )
        self.blocks = {layout.block} if self.enable_blocks else set()
        self.ramp = layout.ramp
        self.low_wall = layout.low_wall
        for visits in self._visits.values():
            visits.fill(0)
        self._last_seen = {"seeker": None, "hider": None}
        self.last_events = []
        _, info = super().reset(seed=None, options=None)
        if options is not None:
            if set(options) != {"seeker_pos", "hider_pos"}:
                raise ValueError("reset options require seeker_pos and hider_pos")
            seeker_pos = options["seeker_pos"]
            hider_pos = options["hider_pos"]
            if not (
                isinstance(seeker_pos, tuple) and len(seeker_pos) == 2
                and all(isinstance(value, int) for value in seeker_pos)
                and isinstance(hider_pos, tuple) and len(hider_pos) == 2
                and all(isinstance(value, int) for value in hider_pos)
                and self._inside(seeker_pos) and self._inside(hider_pos)
                and seeker_pos != hider_pos
            ):
                raise ValueError("fixed spawn positions must be distinct free cells")
            self.seeker_pos, self.hider_pos = seeker_pos, hider_pos
            info["distance"] = self._reported_distance()
        if self.opponent_policies is not None:
            self.opponent_index = int(self.np_random.integers(len(self.opponent_policies)))
            self.opponent_policy = self.opponent_policies[self.opponent_index]
        self._update_history()
        info["layout_index"] = self.layout_index
        info["opponent_index"] = self.opponent_index
        return self._observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
        seen_before = self.is_visible("seeker")
        self.last_events = []
        _, reward, terminated, truncated, info = HideSeekEnv.step(self, action)
        new_cell = self._visits["seeker"][
            self.seeker_pos[1] * self.grid_size + self.seeker_pos[0]
        ] == 0
        self._update_history()
        if self.role == "seeker" and new_cell and not seen_before and not (terminated or truncated):
            reward += self.novelty_bonus
        info["layout_index"] = self.layout_index
        info["opponent_index"] = self.opponent_index
        return self._observation(), reward, terminated, truncated, info

    def _spawn_cells(self) -> tuple[int, ...]:
        return tuple(
            cell for cell in self._free_cells
            if (cell % self.grid_size, cell // self.grid_size) not in self.blocks
        )

    def _inside(self, position: Cell) -> bool:
        return super()._inside(position) and position not in self.blocks

    def _is_wall(self, position: Cell) -> bool:
        return super()._is_wall(position) or position in self.blocks

    def _move(self, position: Cell, action: int) -> Cell:
        dx, dy = self._ACTION_DELTAS[action]
        target = (position[0] + dx, position[1] + dy)
        actor = "seeker" if position == self.seeker_pos else "hider"
        if target in self.blocks:
            beyond = (target[0] + dx, target[1] + dy)
            other = self.hider_pos if position == self.seeker_pos else self.seeker_pos
            if super()._inside(beyond) and beyond not in self.blocks and beyond != other:
                self.blocks.remove(target)
                self.blocks.add(beyond)
                self.last_events.append((actor, "block_pushed"))
                return target
            return position
        if self.enable_ramp and target == self.low_wall and position == self.ramp:
            landing = (target[0] + dx, target[1] + dy)
            if self._inside(landing):
                self.last_events.append((actor, "climbed"))
                return landing
        return target if self._inside(target) else position

    def _observation_for(self, role: Literal["seeker", "hider"]) -> np.ndarray:
        base = super()._observation_for(role)
        size = self.grid_size

        def layer(cells: set[Cell]) -> np.ndarray:
            result = np.zeros(size * size, dtype=np.float32)
            for x, y in cells:
                result[y * size + x] = 1.0
            return result

        last = self._last_seen[role]
        last_values = np.asarray(
            (0.0, 0.0, 0.0) if last is None else (
                last[0] / (size - 1), last[1] / (size - 1), 1.0
            ), dtype=np.float32,
        )
        return np.concatenate((
            base,
            layer(self.blocks),
            layer({self.ramp} if self.enable_ramp else set()),
            layer({self.low_wall} if self.enable_ramp else set()),
            self._visits[role],
            last_values,
        ))

    def _update_history(self) -> None:
        for role, position, opponent in (
            ("seeker", self.seeker_pos, self.hider_pos),
            ("hider", self.hider_pos, self.seeker_pos),
        ):
            self._visits[role][position[1] * self.grid_size + position[0]] = 1.0
            if self._line_of_sight(position, opponent):
                self._last_seen[role] = opponent

    def _paint_obstacles(self, image: np.ndarray, cell_size: int, margin: int) -> None:
        super()._paint_obstacles(image, cell_size, margin)
        for cells, color in (({self.low_wall} if self.enable_ramp else set(), (170, 105, 40)),
                             ({self.ramp} if self.enable_ramp else set(), (230, 175, 55)),
                             (self.blocks, (115, 70, 40))):
            for x, y in cells:
                image[
                    y * cell_size + margin : (y + 1) * cell_size - margin,
                    x * cell_size + margin : (x + 1) * cell_size - margin,
                    :,
                ] = color
