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


def simulate_additive_ou_process(
    n_paths: int,
    episode_length: int,
    speed_of_mean_reversion: float = 1.0,
    long_term_mean: float = 0.0,
    volatility: float = 1.2,
    start_value: float = 0.0,
    time_step: float = 1.0 / 365.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulates an exactly discretized additive Ornstein-Uhlenbeck process.

    Args:
        n_paths: Number of paths to simulate.
        episode_length: Number of daily observations per path.
        speed_of_mean_reversion: Positive pull toward the long-term mean.
        long_term_mean: Stationary mean of the process.
        volatility: Diffusion volatility in price units.
        start_value: Initial process value for every path.
        time_step: Duration between two observations.
        seed: Random seed.

    Returns:
        Process values with shape ``(n_paths, episode_length)``.
    """
    if speed_of_mean_reversion <= 0.0:
        raise ValueError("speed_of_mean_reversion must be positive")
    if time_step <= 0.0:
        raise ValueError("time_step must be positive")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")

    rng = np.random.default_rng(seed)
    process = np.full((n_paths, episode_length), start_value, dtype=np.float64)
    if episode_length <= 1:
        return process
    decay = np.exp(-speed_of_mean_reversion * time_step)
    innovation_std = volatility * np.sqrt(
        (1.0 - decay**2) / (2.0 * speed_of_mean_reversion)
    )
    shocks = rng.normal(0.0, innovation_std, size=(n_paths, episode_length - 1))
    for step in range(1, episode_length):
        process[:, step] = (
            long_term_mean
            + decay * (process[:, step - 1] - long_term_mean)
            + shocks[:, step - 1]
        )
    return process
