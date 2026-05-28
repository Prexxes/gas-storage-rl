"""Ornstein-Uhlenbeck residual simulation."""

from __future__ import annotations

import numpy as np


def simulate_ou_residuals(
    n_paths: int,
    episode_length: int,
    mean_reversion: float = 0.08,
    volatility: float = 0.12,
    seed: int | None = None,
) -> np.ndarray:
    """Simulates mean-reverting residuals around zero.

    Args:
        n_paths: Number of paths.
        episode_length: Number of time steps.
        mean_reversion: Daily OU pull toward zero.
        volatility: Daily residual volatility.
        seed: Random seed.

    Returns:
        Residual matrix with shape ``(n_paths, episode_length)``.
    """
    rng = np.random.default_rng(seed)
    residuals = np.zeros((n_paths, episode_length), dtype=np.float64)
    shocks = rng.normal(0.0, volatility, size=(n_paths, episode_length - 1))
    for step in range(1, episode_length):
        residuals[:, step] = (
            (1.0 - mean_reversion) * residuals[:, step - 1] + shocks[:, step - 1]
        )
    return residuals
