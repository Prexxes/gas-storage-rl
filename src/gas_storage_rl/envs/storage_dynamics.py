"""Storage inventory dynamics and action clipping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StorageParams:
    """Physical gas storage parameters for the MVP model."""

    capacity: float
    injection_rate: float = 1.0
    withdrawal_rate: float = 1.0
    initial_inventory: float = 0.0
    target_terminal_inventory: float = 0.0
    efficiency: float = 1.0
    transaction_cost: float = 0.0
    leakage: float = 0.0


def clip_storage_action(
    requested_action: float,
    storage_level: float,
    params: StorageParams,
) -> float:
    """Clips an action to rate and inventory constraints.

    Args:
        requested_action: Positive values inject gas, negative values withdraw gas.
        storage_level: Current storage inventory.
        params: Storage parameter object.

    Returns:
        The feasible executed action.
    """
    lower_bound = max(-params.withdrawal_rate, -storage_level)
    upper_bound = min(params.injection_rate, params.capacity - storage_level)
    return float(min(max(requested_action, lower_bound), upper_bound))
