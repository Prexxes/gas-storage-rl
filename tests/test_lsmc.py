"""Tests for LSMC benchmark."""

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
