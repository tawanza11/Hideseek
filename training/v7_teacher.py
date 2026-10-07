"""Observable-state search teacher used as a baseline and for imitation data."""

from __future__ import annotations

from collections import deque
from math import isqrt

import numpy as np


DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _cells(values: np.ndarray, size: int) -> set[tuple[int, int]]:
    return {
        (index % size, index // size)
        for index, value in enumerate(values)
        if value > 0.5
    }


def legal_actions(observation: np.ndarray) -> tuple[int, ...]:
    """Physical actions inferred without the hidden opponent position."""
    size = isqrt((len(observation) - 8) // 5)
    if 8 + 5 * size * size != len(observation) or size < 2:
        raise ValueError("V7 observation shape is invalid")
    area = size * size
    walls = _cells(observation[5 : 5 + area], size)
    blocks = _cells(observation[5 + area : 5 + 2 * area], size)
    ramps = _cells(observation[5 + 2 * area : 5 + 3 * area], size)
    low_walls = _cells(observation[5 + 3 * area : 5 + 4 * area], size)
    x = round(float(observation[0]) * (size - 1))
    y = round(float(observation[1]) * (size - 1))

    def floor(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < size and 0 <= cell[1] < size and cell not in walls

    choices = []
    for action, (dx, dy) in enumerate(DIRECTIONS):
        adjacent = (x + dx, y + dy)
        beyond = (adjacent[0] + dx, adjacent[1] + dy)
        if adjacent in blocks:
            allowed = floor(beyond) and beyond not in blocks
        elif adjacent in low_walls:
            allowed = (x, y) in ramps and floor(beyond) and beyond not in blocks
        else:
            allowed = floor(adjacent)
        if allowed:
            choices.append(action)
    return tuple(choices)


def search_recommendations(observation: np.ndarray) -> tuple[int | None, int | None]:
    """Shortest-route actions for chase and exploration from the visible state.

    Only the role's observation is read. No environment state is consulted.
    """
    size = isqrt((len(observation) - 8) // 5)
    if 8 + 5 * size * size != len(observation) or size < 2:
        raise ValueError("V7 observation shape is invalid")
    area = size * size
    walls = _cells(observation[5 : 5 + area], size)
    blocks = _cells(observation[5 + area : 5 + 2 * area], size)
    ramps = _cells(observation[5 + 2 * area : 5 + 3 * area], size)
    low_walls = _cells(observation[5 + 3 * area : 5 + 4 * area], size)
    visited = _cells(observation[5 + 4 * area : 5 + 5 * area], size)
    position = (
        round(float(observation[0]) * (size - 1)),
        round(float(observation[1]) * (size - 1)),
    )

    def floor(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < size and 0 <= cell[1] < size and cell not in walls

    queue = deque([position])
    paths: dict[tuple[int, int], tuple[int, ...]] = {position: ()}
    while queue:
        current = queue.popleft()
        for action, (dx, dy) in enumerate(DIRECTIONS):
            adjacent = (current[0] + dx, current[1] + dy)
            destination = adjacent
            if adjacent in blocks:
                beyond = (adjacent[0] + dx, adjacent[1] + dy)
                if not floor(beyond) or beyond in blocks:
                    continue
            elif adjacent in low_walls and current in ramps:
                destination = (adjacent[0] + dx, adjacent[1] + dy)
                if not floor(destination) or destination in blocks:
                    continue
            elif not floor(adjacent):
                continue
            if destination not in paths:
                paths[destination] = paths[current] + (action,)
                queue.append(destination)

    target: tuple[int, int] | None = None
    if observation[4] > 0.5:
        target = (
            round(float(observation[2]) * (size - 1)),
            round(float(observation[3]) * (size - 1)),
        )
    elif observation[-1] > 0.5:
        target = (
            round(float(observation[-3]) * (size - 1)),
            round(float(observation[-2]) * (size - 1)),
        )
    chase = paths[target][0] if target in paths and target != position else None

    unseen = [cell for cell in paths if cell not in visited and cell != position]
    if unseen:
        target = min(unseen, key=lambda cell: (len(paths[cell]), cell[1], cell[0]))
        return chase, paths[target][0]

    # Explore a distant reachable cell if every reachable floor was visited.
    alternatives = [cell for cell in paths if cell != position]
    if alternatives:
        target = max(alternatives, key=lambda cell: (len(paths[cell]), -cell[1], -cell[0]))
        return chase, paths[target][0]
    return chase, None


def search_features(observation: np.ndarray) -> np.ndarray:
    """Eight planner hints: four chase directions and four explore directions."""
    chase, explore = search_recommendations(observation)
    features = np.zeros(8, dtype=np.float32)
    if chase is not None:
        features[chase] = 1.0
    if explore is not None:
        features[4 + explore] = 1.0
    return features


def search_action(observation: np.ndarray) -> int:
    """Scripted comparison policy; use chase hint before explore hint."""
    chase, explore = search_recommendations(observation)
    return chase if chase is not None else explore if explore is not None else 0
