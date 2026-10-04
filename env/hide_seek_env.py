"""A small grid based Hide-and-Seek environment for either learner role."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HideSeekEnv(gym.Env[np.ndarray, int]):
    """Train one agent against a fixed opponent policy on an open grid.

    Coordinates use ``(x, y)`` with ``(0, 0)`` at the upper-left corner.
    Actions are ordered up, down, left, right. Boundary attempts leave an
    agent in its current cell. There are no interior walls in this V1.

    The observation contains normalized seeker and hider coordinates followed
    by the seeker's normalized clearance to the left, right, top, and bottom
    boundaries.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    _ACTION_DELTAS: tuple[tuple[int, int], ...] = (
        (0, -1),  # up
        (0, 1),   # down
        (-1, 0),  # left
        (1, 0),   # right
    )

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 64,
        render_mode: str | None = None,
        step_cost: float = -0.01,
        role: Literal["seeker", "hider"] = "seeker",
        opponent_policy: Callable[[np.ndarray], int] | None = None,
    ) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if step_cost > 0:
            raise ValueError("step_cost must be zero or negative")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if role not in ("seeker", "hider"):
            raise ValueError("role must be 'seeker' or 'hider'")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_cost = float(step_cost)
        self.role = role
        self.opponent_policy = opponent_policy
        self.action_space = spaces.Discrete(len(self._ACTION_DELTAS))
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )
        self.seeker_pos: tuple[int, int] = (0, 0)
        self.hider_pos: tuple[int, int] = (1, 0)
        self.steps = 0
        self._captured = False
        self._pygame: Any | None = None
        self._screen: Any | None = None
        self._clock: Any | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, bool | int]]:
        super().reset(seed=seed)
        del options  # Reserved for future reset customization.

        spawn_cells = np.asarray(self._spawn_cells(), dtype=np.int64)
        cells = self.np_random.permutation(spawn_cells)
        seeker_cell, hider_cell = int(cells[0]), int(cells[1])
        self.seeker_pos = (seeker_cell % self.grid_size, seeker_cell // self.grid_size)
        self.hider_pos = (hider_cell % self.grid_size, hider_cell // self.grid_size)
        self.steps = 0
        self._captured = False
        return self._observation(), {
            "success": False,
            "captured": False,
            "distance": self._reported_distance(),
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, bool | int]]:
        if self._captured or self.steps >= self.max_steps:
            raise RuntimeError("step called after the episode ended; call reset() first")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, 3], got {action!r}")

        self.steps += 1
        if self.role == "seeker":
            # Both policies observe the state at the start of the turn.
            # The hider's move is still resolved after the seeker's move.
            hider_action = (
                self._opponent_action("hider") if self.opponent_policy is not None else None
            )
            self.seeker_pos = self._move(self.seeker_pos, int(action))
            captured = self.seeker_pos == self.hider_pos
            if not captured:
                if hider_action is None:
                    hider_action = self._opponent_action("hider")
                self.hider_pos = self._move(self.hider_pos, hider_action)
                captured = self.seeker_pos == self.hider_pos
        else:
            seeker_action = self._opponent_action("seeker")
            self.seeker_pos = self._move(self.seeker_pos, seeker_action)
            captured = self.seeker_pos == self.hider_pos
            if not captured:
                self.hider_pos = self._move(self.hider_pos, int(action))
                captured = self.seeker_pos == self.hider_pos

        self._captured = captured
        terminated = captured
        truncated = not captured and self.steps >= self.max_steps
        if self.role == "seeker":
            reward = 1.0 if captured else -1.0 if truncated else self.step_cost
            success = captured
        else:
            reward = -1.0 if captured else 1.0 if truncated else -self.step_cost
            success = truncated

        info: dict[str, bool | int] = {
            "success": success,
            "captured": captured,
            "distance": self._reported_distance(),
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        cell_size = 48
        margin = 2
        image = np.full(
            (self.grid_size * cell_size, self.grid_size * cell_size, 3),
            (245, 245, 245),
            dtype=np.uint8,
        )
        for coordinate in range(1, self.grid_size):
            line = coordinate * cell_size
            image[line - 1 : line + 1, :, :] = (205, 205, 205)
            image[:, line - 1 : line + 1, :] = (205, 205, 205)
        self._paint_obstacles(image, cell_size, margin)
        self._paint_cell(image, self.hider_pos, cell_size, margin, (70, 180, 90))
        self._paint_cell(image, self.seeker_pos, cell_size, margin, (65, 115, 220))

        if self.render_mode == "rgb_array":
            return image

        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("human rendering requires pygame") from error

        if self._pygame is None:
            pygame.init()
            self._screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
            pygame.display.set_caption("Hide and Seek RL")
            self._clock = pygame.time.Clock()
            self._pygame = pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise KeyboardInterrupt("Viewer closed")
        surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None

    def _observation(self) -> np.ndarray:
        return self._observation_for(self.role)

    def _observation_for(self, role: Literal["seeker", "hider"]) -> np.ndarray:
        denominator = float(self.grid_size - 1)
        controlled_pos, opponent_pos = (
            (self.seeker_pos, self.hider_pos)
            if role == "seeker"
            else (self.hider_pos, self.seeker_pos)
        )
        controlled_x, controlled_y = controlled_pos
        opponent_x, opponent_y = opponent_pos
        values = (
            controlled_x / denominator,
            controlled_y / denominator,
            opponent_x / denominator,
            opponent_y / denominator,
            controlled_x / denominator,
            (self.grid_size - 1 - controlled_x) / denominator,
            controlled_y / denominator,
            (self.grid_size - 1 - controlled_y) / denominator,
        )
        return np.asarray(values, dtype=np.float32)

    def _opponent_action(self, opponent_role: Literal["seeker", "hider"]) -> int:
        if self.opponent_policy is not None:
            action = self.opponent_policy(self._observation_for(opponent_role))
            if not self.action_space.contains(action):
                raise ValueError(f"opponent policy action must be an integer in [0, 3], got {action!r}")
            return int(action)

        if opponent_role == "seeker":
            return int(self.np_random.integers(len(self._ACTION_DELTAS)))

        legal_hider_actions = [
            index
            for index, (dx, dy) in enumerate(self._ACTION_DELTAS)
            if self._inside((self.hider_pos[0] + dx, self.hider_pos[1] + dy))
        ]
        return int(self.np_random.choice(legal_hider_actions))

    def _move(self, position: tuple[int, int], action: int) -> tuple[int, int]:
        dx, dy = self._ACTION_DELTAS[action]
        candidate = (position[0] + dx, position[1] + dy)
        return candidate if self._inside(candidate) else position

    def _inside(self, position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _spawn_cells(self) -> tuple[int, ...]:
        """Return row-major cell IDs available when an episode starts."""
        return tuple(range(self.grid_size * self.grid_size))

    def _paint_obstacles(self, image: np.ndarray, cell_size: int, margin: int) -> None:
        """Allow grid environments to paint obstacles before the players."""
        del image, cell_size, margin

    def _reported_distance(self) -> int:
        """Distance included in info; subclasses may hide it from an agent."""
        return self._distance()

    def _distance(self) -> int:
        return abs(self.seeker_pos[0] - self.hider_pos[0]) + abs(
            self.seeker_pos[1] - self.hider_pos[1]
        )

    @staticmethod
    def _paint_cell(
        image: np.ndarray,
        position: tuple[int, int],
        cell_size: int,
        margin: int,
        color: tuple[int, int, int],
    ) -> None:
        x, y = position
        image[
            y * cell_size + margin : (y + 1) * cell_size - margin,
            x * cell_size + margin : (x + 1) * cell_size - margin,
            :,
        ] = color
