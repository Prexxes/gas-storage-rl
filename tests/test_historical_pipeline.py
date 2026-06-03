"""Tests for historical calibration and backtest dataset generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gas_storage_rl.data.path_dataset import (
    build_historical_backtest_paths,
    load_or_generate_historical_backtest_dataset,
)
from gas_storage_rl.prices.calibration import calibrate_historical_price_process
from gas_storage_rl.prices.generators import (
    calibrated_monthly_log_curve,
    generate_price_paths,
)
from gas_storage_rl.prices.historical_data import load_historical_price_csv
from gas_storage_rl.prices.seasonal import fit_monthly_log_seasonality


def test_load_historical_price_csv_infers_price_column(tmp_path) -> None:
    """Historical CSV loading validates dates, split, and positive prices."""
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "date,the_day_ahead_eur_mwh,split\n"
        "2024-01-02,40.0,calibration\n"
        "2024-01-01,41.0,calibration\n",
        encoding="utf-8",
    )

    series = load_historical_price_csv(csv_path, expected_split="calibration")

    assert series.price_column == "the_day_ahead_eur_mwh"
    assert list(series.prices) == [41.0, 40.0]


def test_monthly_log_seasonality_has_zero_mean_adjustments(tmp_path) -> None:
    """Monthly calibration estimates one zero-mean adjustment per month."""
    csv_path = _write_monthly_csv(tmp_path)

    series = load_historical_price_csv(csv_path, expected_split="calibration")
    seasonality = fit_monthly_log_seasonality(series)

    assert len(seasonality.monthly_log_adjustments) == 12
    assert np.isclose(np.mean(seasonality.monthly_log_adjustments), 0.0)


def test_historical_calibration_excludes_post_cutoff_data(tmp_path) -> None:
    """Calibration rejects daily data beyond the configured cutoff."""
    monthly_csv = _write_monthly_csv(tmp_path)
    daily_csv = tmp_path / "daily.csv"
    daily_csv.write_text(
        "date,the_day_ahead_eur_mwh,split\n"
        "2024-12-30,50.0,calibration\n"
        "2025-01-01,51.0,calibration\n",
        encoding="utf-8",
    )

    try:
        calibrate_historical_price_process(
            monthly_csv,
            daily_csv,
            calibration_end_date="2024-12-31",
        )
    except ValueError as exc:
        assert "after calibration cutoff" in str(exc)
    else:
        raise AssertionError("Expected calibration cutoff validation to fail")


def test_calibrated_generator_produces_positive_splits(tmp_path) -> None:
    """Calibrated parameters can drive synthetic positive path generation."""
    monthly_csv = _write_monthly_csv(tmp_path)
    daily_csv = _write_daily_calibration_csv(tmp_path)

    calibration = calibrate_historical_price_process(monthly_csv, daily_csv)
    paths = generate_price_paths(
        "historical_calibrated",
        n_paths=3,
        episode_length=20,
        seed=7,
        params=calibration.to_price_params(),
    )

    assert paths.shape == (3, 20)
    assert np.all(paths > 0.0)
    assert np.allclose(
        paths,
        generate_price_paths(
            "historical_calibrated",
            n_paths=3,
            episode_length=20,
            seed=7,
            params=calibration.to_price_params(),
        ),
    )


def test_historical_environment_variants_use_calibrated_components(tmp_path) -> None:
    """Historical environment names compose seasonality, OU noise, and jumps."""
    monthly_csv = _write_monthly_csv(tmp_path)
    daily_csv = _write_daily_calibration_csv(tmp_path)
    calibration = calibrate_historical_price_process(monthly_csv, daily_csv)
    params = calibration.to_price_params()

    deterministic = generate_price_paths(
        "historical_deterministic",
        n_paths=3,
        episode_length=20,
        seed=7,
        params=params,
    )
    ou_paths = generate_price_paths(
        "historical_ou",
        n_paths=3,
        episode_length=20,
        seed=7,
        params=params,
    )
    jump_paths = generate_price_paths(
        "historical_jump",
        n_paths=3,
        episode_length=20,
        seed=7,
        params={**params, "jump_probability": 1.0, "jump_mean": 0.2, "jump_std": 0.0},
    )

    assert np.allclose(deterministic[0], deterministic[1])
    assert not np.allclose(ou_paths[0], ou_paths[1])
    assert np.all(jump_paths > 0.0)
    assert not np.allclose(jump_paths, ou_paths)


def test_historical_seasonality_uses_smooth_fourier_curve_by_default() -> None:
    """Fourier seasonality avoids hard monthly jumps in daily curves."""
    monthly_adjustments = [0.5, -0.5, 0.2, 0.1, 0.0, -0.1, -0.2, 0.1, 0.2, 0.0, -0.1, -0.2]

    step_curve = calibrated_monthly_log_curve(
        episode_length=40,
        base_log_price=0.0,
        monthly_log_seasonality=monthly_adjustments,
        method="step",
    )
    fourier_curve = calibrated_monthly_log_curve(
        episode_length=40,
        base_log_price=0.0,
        monthly_log_seasonality=monthly_adjustments,
    )

    step_month_change = abs(step_curve[31] - step_curve[30])
    fourier_month_change = abs(fourier_curve[31] - fourier_curve[30])
    assert fourier_month_change < step_month_change


def test_historical_backtest_windows_respect_start_date_and_cache(tmp_path) -> None:
    """Backtest windows are generated separately from held-out dates."""
    csv_path = tmp_path / "backtest.csv"
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=6),
            "the_day_ahead_eur_mwh": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
            "split": "historical_backtest",
        }
    )
    frame.to_csv(csv_path, index=False)

    paths, date_ranges = build_historical_backtest_paths(
        csv_path,
        episode_length=3,
        window_stride=2,
        backtest_start_date="2025-01-01",
    )
    dataset = load_or_generate_historical_backtest_dataset(
        csv_path,
        episode_length=3,
        cache_dir=tmp_path / "cache",
        window_stride=2,
        backtest_start_date="2025-01-01",
    )

    assert paths.shape == (2, 3)
    assert date_ranges[0]["start_date"] == "2025-01-02"
    assert dataset.get_paths("backtest").shape == (2, 3)
    assert dataset.date_ranges_by_split["backtest"][1]["start_date"] == "2025-01-06"


def _write_monthly_csv(tmp_path) -> Path:
    """Writes two years of simple monthly calibration prices."""
    csv_path = tmp_path / "monthly.csv"
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    prices = 50.0 + 5.0 * np.cos(2.0 * np.pi * (dates.month.to_numpy() - 1) / 12.0)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ngc_the_eur_mwh": prices,
            "split": "calibration",
        }
    )
    frame.to_csv(csv_path, index=False)
    return csv_path


def _write_daily_calibration_csv(tmp_path) -> Path:
    """Writes deterministic daily calibration prices through 2024."""
    csv_path = tmp_path / "daily.csv"
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    trend = np.linspace(-0.05, 0.05, len(dates))
    prices = 50.0 * np.exp(trend)
    frame = pd.DataFrame(
        {
            "date": dates,
            "the_day_ahead_eur_mwh": prices,
            "split": "calibration",
        }
    )
    frame.to_csv(csv_path, index=False)
    return csv_path
