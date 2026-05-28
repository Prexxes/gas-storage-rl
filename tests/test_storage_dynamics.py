"""Tests for storage dynamics."""

from gas_storage_rl.envs.storage_dynamics import StorageParams, clip_storage_action


def test_storage_boundary_clipping() -> None:
    """Actions are clipped by boundaries and rates."""
    params = StorageParams(capacity=2.0)
    assert clip_storage_action(1.0, 2.0, params) == 0.0
    assert clip_storage_action(-1.0, 0.0, params) == 0.0
    assert clip_storage_action(5.0, 1.5, params) == 0.5
    assert clip_storage_action(-5.0, 0.5, params) == -0.5
