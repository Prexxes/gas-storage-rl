"""Tests for storage dynamics."""

from gas_storage_rl.envs.storage_dynamics import (
    StorageParams,
    clip_storage_action,
    clip_storage_action_to_terminal_feasibility,
)


def test_storage_boundary_clipping() -> None:
    """Actions are clipped by boundaries and rates."""
    params = StorageParams(capacity=2.0)
    assert clip_storage_action(1.0, 2.0, params) == 0.0
    assert clip_storage_action(-1.0, 0.0, params) == 0.0
    assert clip_storage_action(5.0, 1.5, params) == 0.5
    assert clip_storage_action(-5.0, 0.5, params) == -0.5


def test_terminal_feasibility_clips_unliquidatable_purchase() -> None:
    """Actions cannot buy gas that cannot be withdrawn before maturity."""
    params = StorageParams(capacity=5.0)

    action = clip_storage_action_to_terminal_feasibility(
        requested_action=1.0,
        storage_level=2.0,
        params=params,
        remaining_steps_after_action=2,
        target_inventory=0.0,
    )

    assert action == 0.0


def test_terminal_feasibility_clips_unrecoverable_sale() -> None:
    """Actions cannot sell gas that cannot be reinjected before maturity."""
    params = StorageParams(capacity=5.0)

    action = clip_storage_action_to_terminal_feasibility(
        requested_action=-1.0,
        storage_level=2.0,
        params=params,
        remaining_steps_after_action=2,
        target_inventory=5.0,
    )

    assert action == 1.0


def test_terminal_feasibility_keeps_reachable_action() -> None:
    """Reachable actions are not changed by terminal-feasibility clipping."""
    params = StorageParams(capacity=5.0)

    action = clip_storage_action_to_terminal_feasibility(
        requested_action=0.5,
        storage_level=2.0,
        params=params,
        remaining_steps_after_action=3,
        target_inventory=2.0,
    )

    assert action == 0.5


def test_terminal_feasibility_still_respects_physical_limits() -> None:
    """Already infeasible states do not cause rate or capacity violations."""
    params = StorageParams(capacity=5.0, withdrawal_rate=0.0)

    action = clip_storage_action_to_terminal_feasibility(
        requested_action=0.0,
        storage_level=4.0,
        params=params,
        remaining_steps_after_action=1,
        target_inventory=0.0,
    )

    assert action == 0.0
