"""Fallback synthetic price parameters for the MVP.

Historical gas data calibration is intentionally out of scope for the MVP.
This module centralizes configurable defaults so future calibration can replace
them without changing downstream code.
"""

from __future__ import annotations


def default_price_parameters() -> dict[str, float]:
    """Returns fallback synthetic price process parameters."""
    return {
        "base_price": 50.0,
        "seasonal_amplitude": 0.25,
        "seasonal_phase": 0.0,
        "ou_mean_reversion": 0.08,
        "ou_volatility": 0.12,
        "jump_probability": 0.02,
        "jump_mean": 0.0,
        "jump_std": 0.35,
        "stress_probability": 0.005,
        "stress_multiplier": 1.5,
    }
