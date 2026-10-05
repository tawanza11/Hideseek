"""V4 Seeker rules with a legal-action mask available during training."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from env.hide_seek_v4_env import HideSeekV4Env
from evaluation.evaluate_v31 import legal_actions_from_observation


class HideSeekV5Env(HideSeekV4Env):
    """Expose only moves derivable from the Seeker's own observation."""

    def __init__(
        self,
        *,
        map_pool: tuple[tuple[str, ...], ...],
        novelty_bonus: float = 0.01,
        max_steps: int = 100,
        opponent_policy: Callable[[np.ndarray], int] | None = None,
    ) -> None:
        super().__init__(
            map_pool=map_pool,
            visit_memory=False,
            novelty_bonus=novelty_bonus,
            max_steps=max_steps,
            opponent_policy=opponent_policy,
        )

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[list(legal_actions_from_observation(self._observation()))] = True
        return mask
