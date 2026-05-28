"""Deterministic seasonal log-price curve."""

from __future__ import annotations

import numpy as np


def seasonal_log_curve(
    episode_length: int,
    base_price: float = 50.0,
    amplitude: float = 0.25,
    phase: float = 0.0,
) -> np.ndarray:
    """Creates a smooth deterministic seasonal log-price curve.

    Args:
        episode_length: Number of decision days.
        base_price: Annual average spot price level.
        amplitude: Seasonal log-amplitude.
        phase: Phase shift in radians.

    Returns:
        Array of seasonal log-prices with shape ``(episode_length,)``.
    """
    days = np.arange(episode_length, dtype=np.float64)
    seasonal = amplitude * np.cos(2.0 * np.pi * days / episode_length + phase)
    return np.log(base_price) + seasonal
