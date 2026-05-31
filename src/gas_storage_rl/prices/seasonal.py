"""Deterministic and calibrated seasonal log-price curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gas_storage_rl.prices.historical_data import HistoricalPriceSeries


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


@dataclass(frozen=True)
class MonthlySeasonality:
    """Log-price seasonality estimated by calendar month."""

    base_log_price: float
    monthly_log_adjustments: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validates monthly adjustment shape."""
        if len(self.monthly_log_adjustments) != 12:
            raise ValueError("monthly_log_adjustments must contain exactly 12 values")

    def log_curve_for_dates(self, dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        """Returns seasonal log-prices for the provided dates."""
        months = pd.DatetimeIndex(dates).month.to_numpy()
        adjustments = np.asarray(self.monthly_log_adjustments, dtype=np.float64)
        return self.base_log_price + adjustments[months - 1]

    def as_params(self) -> dict[str, float | list[float]]:
        """Serializes the seasonality for config metadata."""
        return {
            "base_price": float(np.exp(self.base_log_price)),
            "base_log_price": float(self.base_log_price),
            "monthly_log_seasonality": [
                float(value) for value in self.monthly_log_adjustments
            ],
        }


def fit_monthly_log_seasonality(
    monthly_prices: HistoricalPriceSeries,
) -> MonthlySeasonality:
    """Fits a zero-mean monthly log-price seasonality curve.

    Args:
        monthly_prices: Monthly calibration prices.

    Returns:
        Estimated monthly log seasonality.
    """
    data = monthly_prices.data
    log_prices = np.log(data[monthly_prices.price_column].to_numpy(dtype=np.float64))
    base_log_price = float(np.mean(log_prices))
    by_month = pd.DataFrame(
        {
            "month": data["date"].dt.month.to_numpy(),
            "centered_log_price": log_prices - base_log_price,
        }
    )
    monthly_adjustments = by_month.groupby("month")["centered_log_price"].mean()
    if set(monthly_adjustments.index) != set(range(1, 13)):
        raise ValueError("Monthly calibration data must cover all calendar months")
    values = monthly_adjustments.reindex(range(1, 13)).to_numpy(dtype=np.float64)
    values = values - np.mean(values)
    return MonthlySeasonality(
        base_log_price=base_log_price,
        monthly_log_adjustments=tuple(values),
    )
