"""Tests for LSMC benchmark."""

from datetime import date

import numpy as np

from gas_storage_rl.baselines.lsmc import LSMCBenchmark
from gas_storage_rl.envs.storage_dynamics import StorageParams


def test_lsmc_fitting_and_evaluation_on_tiny_dataset() -> None:
    """LSMC fits and evaluates without using future-perfect optimization."""
    paths = np.array([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]], dtype=np.float32)
    lsmc = LSMCBenchmark(StorageParams(capacity=2.0), lambda_terminal=10.0)
    lsmc.fit(paths)
    metrics = lsmc.evaluate(paths)
    assert "mean_return_raw" in metrics
    assert len(lsmc.coefficients) == 3


def test_lsmc_accepts_path_specific_calendar_dates() -> None:
    """LSMC includes path-specific calendar positions in its regressions."""
    paths = np.array([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]], dtype=np.float32)
    start_dates = [date(2001, 1, 1), date(2001, 7, 1)]
    lsmc = LSMCBenchmark(StorageParams(capacity=2.0), lambda_terminal=10.0)

    lsmc.fit(paths, start_dates)
    metrics = lsmc.evaluate(paths, start_dates)

    assert "mean_return_raw" in metrics
    assert lsmc.coefficients[0].shape == (15,)


def test_lsmc_accepts_path_specific_initial_and_target_inventories() -> None:
    """LSMC fits and evaluates one model across inventory targets."""
    paths = np.array([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]], dtype=np.float32)
    inventories = np.array([0.4, 1.2])
    lsmc = LSMCBenchmark(StorageParams(capacity=2.0), lambda_terminal=10.0)

    lsmc.fit(paths, initial_inventories=inventories)
    metrics = lsmc.evaluate(paths, initial_inventories=inventories)

    assert "mean_return_raw" in metrics
