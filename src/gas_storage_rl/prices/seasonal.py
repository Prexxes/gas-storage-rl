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
    annual_period: int = 365,
) -> np.ndarray:
    """Creates a smooth deterministic seasonal log-price curve.

    Args:
        episode_length: Number of decision days.
        base_price: Annual average spot price level.
        amplitude: Seasonal log-amplitude.
        phase: Phase shift in radians.
        annual_period: Number of days in one seasonal cycle.

    Returns:
        Array of seasonal log-prices with shape ``(episode_length,)``.
    """
    days = np.arange(episode_length, dtype=np.float64)
    seasonal = amplitude * np.cos(2.0 * np.pi * days / annual_period + phase)
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

    def log_curve_for_dates(
        self,
        dates: pd.Series | pd.DatetimeIndex,
        method: str = "fourier",
        fourier_harmonics: int = 2,
    ) -> np.ndarray:
        """Returns seasonal log-prices for the provided dates.

        Args:
            dates: Dates at which to evaluate the seasonal curve.
            method: Either ``fourier`` for a smooth periodic curve or ``step``
                for calendar-month constants.
            fourier_harmonics: Number of Fourier harmonics used for smoothing.

        Returns:
            Seasonal log-price array with one value per input date.
        """
        adjustments = np.asarray(self.monthly_log_adjustments, dtype=np.float64)
        if method == "step":
            months = pd.DatetimeIndex(dates).month.to_numpy()
            return self.base_log_price + adjustments[months - 1]
        if method != "fourier":
            raise ValueError(f"Unknown seasonality method: {method}")
        return self.base_log_price + evaluate_fourier_monthly_seasonality(
            dates,
            adjustments,
            harmonics=fourier_harmonics,
        )

    def as_params(self) -> dict[str, float | list[float]]:
        """Serializes the seasonality for config metadata."""
        return {
            "base_price": float(np.exp(self.base_log_price)),
            "base_log_price": float(self.base_log_price),
            "monthly_log_seasonality": [
                float(value) for value in self.monthly_log_adjustments
            ],
            "seasonality_interpolation": "fourier",
            "seasonality_fourier_harmonics": 2,
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


def evaluate_fourier_monthly_seasonality(
    dates: pd.Series | pd.DatetimeIndex,
    monthly_log_adjustments: np.ndarray,
    harmonics: int = 2,
) -> np.ndarray:
    """Evaluates a smooth periodic Fourier curve fitted to monthly values.

    Args:
        dates: Dates at which to evaluate the seasonal adjustment.
        monthly_log_adjustments: Twelve zero-mean monthly log adjustments.
        harmonics: Number of sine/cosine harmonics.

    Returns:
        Smooth daily seasonal log adjustments.
    """
    if len(monthly_log_adjustments) != 12:
        raise ValueError("monthly_log_adjustments must contain exactly 12 values")
    if harmonics < 1:
        raise ValueError("harmonics must be positive")
    fitted_harmonics = min(int(harmonics), 5)
    month_midpoints = np.asarray(
        [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349],
        dtype=np.float64,
    )
    coefficients = np.linalg.lstsq(
        _fourier_design(month_midpoints / 365.0, fitted_harmonics),
        monthly_log_adjustments,
        rcond=None,
    )[0]
    date_index = pd.DatetimeIndex(dates)
    year_lengths = np.where(date_index.is_leap_year, 366.0, 365.0)
    fractions = (date_index.dayofyear.to_numpy(dtype=np.float64) - 0.5) / year_lengths
    values = _fourier_design(fractions, fitted_harmonics) @ coefficients
    return values - np.mean(values)


def _fourier_design(fractions: np.ndarray, harmonics: int) -> np.ndarray:
    """Builds a Fourier design matrix for annual fractions."""
    columns = [np.ones_like(fractions, dtype=np.float64)]
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * np.pi * harmonic * fractions
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))
    return np.column_stack(columns)
