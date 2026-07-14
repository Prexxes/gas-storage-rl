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


def clip_storage_action_to_terminal_feasibility(
    requested_action: float,
    storage_level: float,
    params: StorageParams,
    remaining_steps_after_action: int,
    target_inventory: float,
) -> float:
    """Clips an action so the terminal inventory remains physically reachable.

    The returned action always respects the normal rate and capacity constraints. When
    the current state is already outside the terminal-feasible corridor and no action can
    restore feasibility in one step, the normal rate/capacity-clipped action is returned.

    Args:
        requested_action: Positive values inject gas, negative values withdraw gas.
        storage_level: Current storage inventory.
        params: Storage parameter object.
        remaining_steps_after_action: Decision steps left after executing the action.
        target_inventory: Desired terminal inventory.

    Returns:
        The executed action after physical and terminal-reachability clipping.

    """
    normal_lower_bound = max(-params.withdrawal_rate, -storage_level)
    normal_upper_bound = min(params.injection_rate, params.capacity - storage_level)

    remaining_steps = max(int(remaining_steps_after_action), 0)
    min_reachable_level = max(
        0.0,
        float(target_inventory) - remaining_steps * params.injection_rate,
    )
    max_reachable_level = min(
        params.capacity,
        float(target_inventory) + remaining_steps * params.withdrawal_rate,
    )
    terminal_lower_bound = min_reachable_level - storage_level
    terminal_upper_bound = max_reachable_level - storage_level

    lower_bound = max(normal_lower_bound, terminal_lower_bound)
    upper_bound = min(normal_upper_bound, terminal_upper_bound)
    if lower_bound > upper_bound:
        return float(min(max(requested_action, normal_lower_bound), normal_upper_bound))
    return float(min(max(requested_action, lower_bound), upper_bound))
