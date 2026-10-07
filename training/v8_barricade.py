"""A visible-state demonstration of pushing a nearby block into a doorway."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from training.v7_teacher import DIRECTIONS, legal_actions


def barricade_direction(observation: np.ndarray) -> int | None:
    """Suggest a legal adjacent push using only the Hider's observation."""
    if observation.shape != (508,):
        raise ValueError("V8 Hider observation must contain 508 values")
    block_indices = np.flatnonzero(observation[105:205] > 0.5)
    if len(block_indices) != 1:
        return None
    block_index = int(block_indices[0])
    block = (block_index % 10, block_index // 10)
    position = (round(float(observation[0]) * 9), round(float(observation[1]) * 9))
    legal = legal_actions(observation)
    for action, (dx, dy) in enumerate(DIRECTIONS):
        if action in legal and (position[0] + dx, position[1] + dy) == block:
            return action
    return None


def demonstration_policy(seed: int) -> Callable[[np.ndarray], int]:
    """Push the doorway block when adjacent, then move legally at random."""
    rng = np.random.default_rng(seed)

    def action(observation: np.ndarray) -> int:
        push = barricade_direction(observation)
        if push is not None:
            return push
        legal = legal_actions(observation)
        return int(rng.choice(legal)) if legal else 0

    return action
