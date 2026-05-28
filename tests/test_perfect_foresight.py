"""Tests for perfect-foresight linear program."""

import numpy as np

from gas_storage_rl.baselines.perfect_foresight import PerfectForesightBaseline
from gas_storage_rl.envs.storage_dynamics import StorageParams


def test_perfect_foresight_lp_feasible_and_respects_constraints() -> None:
    """LP returns feasible storage and action paths."""
    baseline = PerfectForesightBaseline(StorageParams(capacity=2.0), lambda_terminal=10.0)
    result = baseline.solve_path(np.array([10.0, 20.0, 30.0]))
    assert result.success
    assert np.all(result.actions <= 1.0 + 1e-8)
    assert np.all(result.actions >= -1.0 - 1e-8)
    assert np.all(result.storage_levels <= 2.0 + 1e-8)
    assert np.all(result.storage_levels >= -1e-8)
