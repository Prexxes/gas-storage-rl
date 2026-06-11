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


def test_perfect_foresight_uses_episode_initial_and_target_inventory() -> None:
    """LP boundary conditions are specific to the evaluated episode."""
    baseline = PerfectForesightBaseline(StorageParams(capacity=2.0), lambda_terminal=100.0)

    result = baseline.solve_path(
        np.array([100.0, 10.0]),
        initial_inventory=1.0,
        target_inventory=1.0,
    )

    assert result.success
    assert np.isclose(result.storage_levels[0], 1.0)
    assert np.isclose(result.storage_levels[-1], 1.0)


def test_perfect_foresight_defaults_target_to_initial_inventory() -> None:
    """LP defaults to ending at the initial inventory, not storage param target."""
    baseline = PerfectForesightBaseline(
        StorageParams(
            capacity=2.0,
            initial_inventory=1.0,
            target_terminal_inventory=0.0,
        ),
        lambda_terminal=100.0,
    )

    result = baseline.solve_path(np.array([100.0, 10.0]))

    assert result.success
    assert np.isclose(result.storage_levels[0], 1.0)
    assert np.isclose(result.storage_levels[-1], 1.0)


def test_perfect_foresight_respects_explicit_target_inventory() -> None:
    """LP still supports an explicitly requested terminal inventory."""
    baseline = PerfectForesightBaseline(
        StorageParams(capacity=2.0, initial_inventory=1.0),
        lambda_terminal=100.0,
    )

    result = baseline.solve_path(
        np.array([100.0, 10.0]),
        target_inventory=0.0,
    )

    assert result.success
    assert np.isclose(result.storage_levels[0], 1.0)
    assert np.isclose(result.storage_levels[-1], 0.0)
