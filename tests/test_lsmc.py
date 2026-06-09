"""Tests for LSMC benchmark."""

from datetime import date

import numpy as np
import pytest

from gas_storage_rl.baselines.lsmc import LSMCBenchmark, _clip_actions
from gas_storage_rl.envs.storage_dynamics import (
    StorageParams,
    clip_storage_action_to_terminal_feasibility,
)


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


def test_lsmc_predict_action_preserves_terminal_reachability() -> None:
    """LSMC returns the terminal-feasible action it evaluates internally."""
    lsmc = LSMCBenchmark(
        StorageParams(
            capacity=5.0,
            injection_rate=1.0,
            withdrawal_rate=1.0,
        ),
        lambda_terminal=10.0,
        action_grid=np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
    )
    lsmc.coefficients[1] = np.zeros(15)

    action = lsmc.predict_action(
        storage_level=3.0,
        price=100.0,
        current_step=1,
        horizon=2,
        target_inventory=3.0,
    )

    assert action == 0.0


def test_lsmc_requires_at_least_two_inventory_levels() -> None:
    """LSMC inventory-grid fitting requires a non-degenerate grid."""
    paths = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    lsmc = LSMCBenchmark(
        StorageParams(capacity=2.0),
        lambda_terminal=10.0,
        n_inventory_levels=1,
    )

    with pytest.raises(ValueError, match="n_inventory_levels"):
        lsmc.fit(paths)


def test_lsmc_vectorized_clipping_matches_environment_scalar_clipping() -> None:
    """Vectorized LSMC fit clipping matches the environment action clip."""
    params = StorageParams(capacity=5.0, injection_rate=1.0, withdrawal_rate=1.0)
    storage = np.array([0.0, 2.0, 4.5])
    targets = np.array([1.0, 2.0, 4.0])

    vectorized = _clip_actions(
        1.0,
        storage,
        params,
        remaining_steps_after_action=0,
        target_inventories=targets,
    )
    scalar = np.array(
        [
            clip_storage_action_to_terminal_feasibility(
                requested_action=1.0,
                storage_level=float(level),
                params=params,
                remaining_steps_after_action=0,
                target_inventory=float(target),
            )
            for level, target in zip(storage, targets, strict=True)
        ]
    )

    np.testing.assert_allclose(vectorized, scalar)
