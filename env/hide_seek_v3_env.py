"""Hide-and-Seek environment with rooms, walls, and line-of-sight vision."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Literal

import numpy as np
from gymnasium import spaces

from env.hide_seek_env import HideSeekEnv


DEFAULT_MAP: tuple[str, ...] = (
    "##########",
    "#...#....#",
    "#...#....#",
    "#........#",
    "#...#....#",
    "###.######",
    "#........#",
    "#....#...#",
    "#....#...#",
    "##########",
)


class HideSeekV3Env(HideSeekEnv):
    """V3 grid game where walls block movement and sight.

    Observations contain the controlled player's normalized position, the
    opponent's normalized position when visible (zeros otherwise), a visibility
    flag, and the complete static wall map in row-major order.
    """

    def __init__(
        self,
        map_rows: tuple[str, ...] | None = None,
        max_steps: int = 100,
        render_mode: str | None = None,
        role: Literal["seeker", "hider"] = "seeker",
        opponent_policy: Callable[[np.ndarray], int] | None = None,
    ) -> None:
        selected_map = DEFAULT_MAP if map_rows is None else map_rows
        grid_size = self._validate_map(selected_map)
        self.map_rows = tuple(selected_map)
        self._free_cells = tuple(
            y * grid_size + x
            for y, row in enumerate(self.map_rows)
            for x, cell in enumerate(row)
            if cell == "."
        )
        super().__init__(
            grid_size=grid_size,
            max_steps=max_steps,
            render_mode=render_mode,
            role=role,
            opponent_policy=opponent_policy,
        )
        observation_size = 5 + grid_size * grid_size
        self.observation_space = spaces.Box(
            low=np.zeros(observation_size, dtype=np.float32),
            high=np.ones(observation_size, dtype=np.float32),
            dtype=np.float32,
        )

    @staticmethod
    def _validate_map(map_rows: tuple[str, ...]) -> int:
        if not isinstance(map_rows, tuple) or not map_rows:
            raise ValueError("map_rows must be a non-empty tuple of strings")
        grid_size = len(map_rows)
        if any(not isinstance(row, str) or len(row) != grid_size for row in map_rows):
            raise ValueError("map_rows must form a square map")
        if any(cell not in ".#" for row in map_rows for cell in row):
            raise ValueError("map_rows may contain only '.' and '#' characters")

        free_cells = {
            (x, y)
            for y, row in enumerate(map_rows)
            for x, cell in enumerate(row)
            if cell == "."
        }
        if len(free_cells) < 2:
            raise ValueError("map must contain at least two free cells")
        start = next(iter(free_cells))
        connected = {start}
        pending = deque([start])
        while pending:
            x, y = pending.popleft()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in free_cells and neighbor not in connected:
                    connected.add(neighbor)
                    pending.append(neighbor)
        if connected != free_cells:
            raise ValueError("all free cells must be connected by cardinal movement")
        return grid_size

    def _spawn_cells(self) -> tuple[int, ...]:
        return self._free_cells

    def _inside(self, position: tuple[int, int]) -> bool:
        x, y = position
        return (
            0 <= x < self.grid_size
            and 0 <= y < self.grid_size
            and self.map_rows[y][x] == "."
        )

    def is_visible(self, role: Literal["seeker", "hider"] | None = None) -> bool:
        """Return whether the opposing player can be seen from ``role``'s cell."""
        observer_role = self.role if role is None else role
        observer, target = (
            (self.seeker_pos, self.hider_pos)
            if observer_role == "seeker"
            else (self.hider_pos, self.seeker_pos)
        )
        return self._line_of_sight(observer, target)

    def _line_of_sight(
        self, start: tuple[int, int], target: tuple[int, int]
    ) -> bool:
        """Trace cell centers; exact corner crossings inspect both side cells."""
        x, y = start
        target_x, target_y = target
        dx, dy = target_x - x, target_y - y
        nx, ny = abs(dx), abs(dy)
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        ix = iy = 0
        while ix < nx or iy < ny:
            cross_x = (1 + 2 * ix) * ny
            cross_y = (1 + 2 * iy) * nx
            if cross_x == cross_y:
                side_x, side_y = (x + step_x, y), (x, y + step_y)
                if self._is_wall(side_x) or self._is_wall(side_y):
                    return False
                x += step_x
                y += step_y
                ix += 1
                iy += 1
            elif cross_x < cross_y:
                x += step_x
                ix += 1
            else:
                y += step_y
                iy += 1
            if (x, y) != target and self._is_wall((x, y)):
                return False
        return True

    def _is_wall(self, position: tuple[int, int]) -> bool:
        x, y = position
        return not (0 <= x < self.grid_size and 0 <= y < self.grid_size) or self.map_rows[y][x] == "#"

    def _observation_for(self, role: Literal["seeker", "hider"]) -> np.ndarray:
        controlled_pos, opponent_pos = (
            (self.seeker_pos, self.hider_pos)
            if role == "seeker"
            else (self.hider_pos, self.seeker_pos)
        )
        visible = self._line_of_sight(controlled_pos, opponent_pos)
        denominator = float(self.grid_size - 1)
        map_values = [
            1.0 if cell == "#" else 0.0
            for row in self.map_rows
            for cell in row
        ]
        values = [
            controlled_pos[0] / denominator,
            controlled_pos[1] / denominator,
            opponent_pos[0] / denominator if visible else 0.0,
            opponent_pos[1] / denominator if visible else 0.0,
            1.0 if visible else 0.0,
            *map_values,
        ]
        return np.asarray(values, dtype=np.float32)

    def _reported_distance(self) -> int:
        return self._distance() if self.is_visible(self.role) else -1

    def _paint_obstacles(self, image: np.ndarray, cell_size: int, margin: int) -> None:
        for y, row in enumerate(self.map_rows):
            for x, cell in enumerate(row):
                if cell == "#":
                    image[
                        y * cell_size + margin : (y + 1) * cell_size - margin,
                        x * cell_size + margin : (x + 1) * cell_size - margin,
                        :,
                    ] = (45, 45, 45)
