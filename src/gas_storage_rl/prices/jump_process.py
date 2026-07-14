"""Jump and stress component simulation for gas spot prices."""

from __future__ import annotations

import numpy as np


def simulate_jump_component(
    n_paths: int,
    episode_length: int,
    jump_probability: float = 0.02,
    jump_mean: float = 0.0,
    jump_std: float = 0.35,
    stress_probability: float = 0.005,
    stress_multiplier: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    """Simulates sparse additive jumps in price units.

    Args:
        n_paths: Number of paths.
        episode_length: Number of time steps.
        jump_probability: Daily probability of a regular jump.
        jump_mean: Mean regular jump size in price units.
        jump_std: Standard deviation of regular jump size in price units.
        stress_probability: Daily probability of a larger stress jump.
        stress_multiplier: Multiplier applied to stress jump volatility.
        seed: Random seed.

    Returns:
        Jump matrix with shape ``(n_paths, episode_length)``.

    """
    rng = np.random.default_rng(seed)
    regular_mask = rng.random((n_paths, episode_length)) < jump_probability
    stress_mask = rng.random((n_paths, episode_length)) < stress_probability
    regular = rng.normal(jump_mean, jump_std, size=(n_paths, episode_length))
    stress = rng.normal(
        abs(jump_mean) + stress_multiplier * jump_std,
        stress_multiplier * jump_std,
        size=(n_paths, episode_length),
    )
    return regular_mask * regular + stress_mask * stress
