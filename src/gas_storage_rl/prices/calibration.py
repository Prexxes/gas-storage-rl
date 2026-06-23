"""Fallback and historical gas spot price calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gas_storage_rl.prices.historical_data import (
    HistoricalPriceSeries,
    assert_date_range,
    load_historical_price_csv,
)
from gas_storage_rl.prices.seasonal import (
    MonthlySeasonality,
    fit_monthly_log_seasonality,
)


def default_price_parameters() -> dict[str, float]:
    """Returns fallback parameters for additive synthetic price processes."""
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


@dataclass(frozen=True)
class HistoricalCalibrationResult:
    """Calibrated seasonal, OU/AR(1), and jump process parameters."""

    seasonality: MonthlySeasonality
    ar1_phi: float
    ou_mean_reversion: float
    ou_volatility: float
    jump_probability: float
    jump_mean: float
    jump_std: float
    jump_threshold_sigma: float
    calibration_start_date: str
    calibration_end_date: str
    calibration_cutoff_date: str
    monthly_calibration_csv: str
    daily_calibration_csv: str
    daily_observations: int
    jump_observations: int

    def to_price_params(self) -> dict[str, float | list[float] | str | int]:
        """Serializes calibrated parameters for generators and cache metadata."""
        params: dict[str, float | list[float] | str | int] = {
            **self.seasonality.as_params(),
            "ar1_phi": self.ar1_phi,
            "ou_mean_reversion": self.ou_mean_reversion,
            "ou_volatility": self.ou_volatility,
            "jump_probability": self.jump_probability,
            "jump_mean": self.jump_mean,
            "jump_std": self.jump_std,
            "stress_probability": 0.0,
            "stress_multiplier": 1.0,
            "jump_threshold_sigma": self.jump_threshold_sigma,
            "calibration_start_date": self.calibration_start_date,
            "calibration_end_date": self.calibration_end_date,
            "calibration_cutoff_date": self.calibration_cutoff_date,
            "monthly_calibration_csv": self.monthly_calibration_csv,
            "daily_calibration_csv": self.daily_calibration_csv,
            "daily_observations": self.daily_observations,
            "jump_observations": self.jump_observations,
        }
        return params


def calibrate_historical_price_process(
    monthly_calibration_csv: str | Path,
    daily_calibration_csv: str | Path,
    *,
    calibration_end_date: str = "2024-12-31",
    monthly_price_column: str | None = None,
    daily_price_column: str | None = None,
    jump_threshold_sigma: float = 3.0,
) -> HistoricalCalibrationResult:
    """Calibrates a seasonal AR(1)/OU jump process from historical prices.

    Args:
        monthly_calibration_csv: Monthly calibration price CSV.
        daily_calibration_csv: Daily calibration price CSV.
        calibration_end_date: Inclusive final date allowed in calibration.
        monthly_price_column: Optional monthly price column override.
        daily_price_column: Optional daily price column override.
        jump_threshold_sigma: Robust innovation threshold used to classify jumps.

    Returns:
        Calibrated historical price process.
    """
    monthly_prices = load_historical_price_csv(
        monthly_calibration_csv,
        price_column=monthly_price_column,
        expected_split="calibration",
    )
    daily_prices = load_historical_price_csv(
        daily_calibration_csv,
        price_column=daily_price_column,
        expected_split="calibration",
    )
    assert_date_range(monthly_prices, max_date=calibration_end_date)
    assert_date_range(daily_prices, max_date=calibration_end_date)

    seasonality = fit_monthly_log_seasonality(monthly_prices)
    residuals = compute_daily_residuals(daily_prices, seasonality)
    phi, innovations = _fit_ar1(residuals)
    robust_sigma = _robust_sigma(innovations)
    threshold = jump_threshold_sigma * robust_sigma
    jump_mask = np.abs(innovations) > threshold
    non_jump_innovations = innovations[~jump_mask]
    if len(non_jump_innovations) == 0:
        non_jump_innovations = innovations
    jump_sizes = innovations[jump_mask]

    return HistoricalCalibrationResult(
        seasonality=seasonality,
        ar1_phi=float(phi),
        ou_mean_reversion=float(1.0 - phi),
        ou_volatility=(
            float(np.std(non_jump_innovations, ddof=1))
            if len(non_jump_innovations) > 1
            else 0.0
        ),
        jump_probability=float(np.mean(jump_mask)) if len(jump_mask) else 0.0,
        jump_mean=float(np.mean(jump_sizes)) if len(jump_sizes) else 0.0,
        jump_std=float(np.std(jump_sizes, ddof=1)) if len(jump_sizes) > 1 else 0.0,
        jump_threshold_sigma=float(jump_threshold_sigma),
        calibration_start_date=str(daily_prices.dates.iloc[0].date()),
        calibration_end_date=str(daily_prices.dates.iloc[-1].date()),
        calibration_cutoff_date=calibration_end_date,
        monthly_calibration_csv=str(monthly_calibration_csv),
        daily_calibration_csv=str(daily_calibration_csv),
        daily_observations=int(len(daily_prices.data)),
        jump_observations=int(np.sum(jump_mask)),
    )


def compute_daily_residuals(
    daily_prices: HistoricalPriceSeries,
    seasonality: MonthlySeasonality,
) -> np.ndarray:
    """Computes daily log-price residuals after removing monthly seasonality."""
    log_prices = np.log(daily_prices.prices.to_numpy(dtype=np.float64))
    seasonal_log_prices = seasonality.log_curve_for_dates(daily_prices.dates)
    residuals = log_prices - seasonal_log_prices
    return residuals - np.mean(residuals)


def _fit_ar1(residuals: np.ndarray) -> tuple[float, np.ndarray]:
    """Fits ``r_t = phi * r_{t-1} + epsilon_t`` by least squares."""
    if len(residuals) < 3:
        raise ValueError(
            "At least three daily observations are required for AR(1) calibration"
        )
    previous = residuals[:-1]
    current = residuals[1:]
    denominator = float(np.dot(previous, previous))
    phi = float(np.dot(previous, current) / denominator) if denominator > 0.0 else 0.0
    phi = float(np.clip(phi, -0.99, 0.99))
    innovations = current - phi * previous
    return phi, innovations


def _robust_sigma(values: np.ndarray) -> float:
    """Returns a positive robust scale estimate based on MAD."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    sigma = float(1.4826 * mad)
    if sigma <= 0.0:
        sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return max(sigma, np.finfo(np.float64).eps)
