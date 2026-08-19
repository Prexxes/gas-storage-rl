"""Default synthetic gas spot price process parameters."""

from __future__ import annotations


def default_price_parameters() -> dict[str, float]:
    """Returns fallback parameters for additive synthetic price processes.

    Returns:
        Fallback price process parameters.

    """
    return {
        "seasonal_level": 2.0,
        "seasonal_amplitude": 1.0,
        "seasonal_period": 365.0,
        "ou_speed_of_mean_reversion": 1.0,
        "ou_long_term_mean": 0.0,
        "ou_volatility": 1.2,
        "ou_start_value": 0.0,
        "ou_time_step": 1.0 / 365.0,
        "jump_probability": 0.02,
        "jump_mean": 0.0,
        "jump_std": 0.35,
        "stress_probability": 0.005,
        "stress_multiplier": 1.5,
    }
