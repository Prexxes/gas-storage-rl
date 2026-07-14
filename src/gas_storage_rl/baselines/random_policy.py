"""Uniform random storage policy."""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Samples actions uniformly from [-1, 1]."""

    def __init__(self, seed: int | None = None) -> None:
        """Initializes the policy.
        
        Args:
            seed: Random seed for deterministic behavior.

        """
        self.rng = np.random.default_rng(seed)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Returns a random action compatible with Stable-Baselines3.
        
        Args:
            observation: Observation value.
            deterministic: Deterministic value.
        
        Returns:
            Predicted action and optional recurrent state.

        """
        del observation, deterministic
        return np.array([self.rng.uniform(-1.0, 1.0)], dtype=np.float32), None
