"""Tests for historical CSV loading and backtest dataset generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gas_storage_rl.data.path_dataset import (
    PathDataset,
    build_historical_backtest_paths,
    load_or_generate_historical_backtest_dataset,
)
from gas_storage_rl.envs.storage_dynamics import StorageParams
from gas_storage_rl.evaluation.backtest import (
    build_backtest_evaluation_dataset,
    evaluate_policy_on_backtest,
)
from gas_storage_rl.prices.historical_data import load_historical_price_csv


class ZeroPolicy:
    """Policy that always requests no storage action."""

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Returns the no-op action in the SB3 policy format."""
        del observation, deterministic
        return np.array([0.0], dtype=np.float32), None


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


def test_backtest_dataset_preserves_synthetic_splits(tmp_path) -> None:
    """Backtest augmentation keeps existing synthetic paths and metadata."""
    config = _backtest_config(tmp_path)
    synthetic_dataset = PathDataset(
        {
            "train": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "validation": np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
            "test": np.array([[3.0, 4.0, 5.0]], dtype=np.float32),
        },
        {"train": 1, "validation": 2, "test": 3},
        {
            "train": [{"start_date": "2024-01-01", "end_date": "2024-01-03"}],
            "validation": [{"start_date": "2024-02-01", "end_date": "2024-02-03"}],
            "test": [{"start_date": "2024-03-01", "end_date": "2024-03-03"}],
        },
    )

    dataset = build_backtest_evaluation_dataset(config, synthetic_dataset)

    assert set(dataset.paths_by_split) == {"train", "validation", "test", "backtest"}
    assert dataset.get_paths("train").tolist() == [[1.0, 2.0, 3.0]]
    assert dataset.get_paths("backtest").shape == (2, 3)
    assert dataset.date_ranges_by_split["backtest"][0] == {
        "start_date": "2025-01-02",
        "end_date": "2025-01-06",
    }


def test_evaluate_policy_on_backtest_accepts_any_predict_policy(tmp_path) -> None:
    """Backtest evaluation works for an arbitrary SB3-style policy object."""
    config = _backtest_config(tmp_path)
    synthetic_dataset = PathDataset(
        {"train": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
        {"train": 1},
    )

    metrics, trajectories = evaluate_policy_on_backtest(
        ZeroPolicy(),
        config,
        synthetic_dataset=synthetic_dataset,
        storage_params=StorageParams(capacity=30.0),
        env_kwargs={"price_scale": 50.0, "reward_scale": 50.0, "seed": 1},
    )

    assert metrics["split"] == "backtest"
    assert len(trajectories) == 2


def _write_backtest_csv(tmp_path: Path) -> Path:
    """Writes a tiny held-out historical backtest CSV."""
    csv_path = tmp_path / "backtest.csv"
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=4),
            "the_day_ahead_eur_mwh": [50.0, 51.0, 52.0, 53.0],
            "split": "historical_backtest",
        }
    )
    frame.to_csv(csv_path, index=False)
    return csv_path


def _backtest_config(tmp_path: Path) -> dict:
    """Returns a minimal config with synthetic training and backtest data."""
    backtest_csv = _write_backtest_csv(tmp_path)
    return {
        "environment_config": {
            "environment_name": "deterministic",
            "capacity": 30,
            "episode_length": 3,
            "initial_inventory": 0.0,
            "initial_inventory_mean_fraction": 0.30,
            "initial_inventory_std_fraction": 0.05,
        },
        "dataset_config": {
            "n_train_paths": 1,
            "n_validation_paths": 1,
            "n_test_paths": 1,
            "cache_dir": str(tmp_path / "cache"),
            "backtest_cache_dir": str(tmp_path / "backtest-cache"),
            "use_cache": True,
            "force_regenerate": False,
        },
        "backtest_data_config": {
            "daily_backtest_csv": str(backtest_csv),
            "backtest_start_date": "2025-01-01",
            "window_stride": 1,
        },
        "price_process_config": {
            "seasonal_level": 2.0,
            "seasonal_amplitude": 1.0,
            "seasonal_period": 365.0,
        },
        "seeds": {"dataset_seed": 1, "eval_seed": 2},
    }


def test_retained_calibration_split_csv_files_exist() -> None:
    """Prepared calibration CSV files are retained as project data."""
    monthly_csv = Path("data/gas_price_data_splits/monthly_calibration_2016_2024.csv")
    daily_csv = Path(
        "data/gas_price_data_splits/daily_calibration_calendar_ffill_2022_2024.csv"
    )

    assert monthly_csv.exists()
    assert daily_csv.exists()
