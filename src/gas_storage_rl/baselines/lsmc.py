"""Least-Squares Monte Carlo benchmark with a discrete action grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from gas_storage_rl.envs.storage_dynamics import (
    StorageParams,
    clip_storage_action_to_terminal_feasibility,
)


def _features(
    storage: np.ndarray,
    price: np.ndarray,
    time_fraction: float,
    capacity: float,
    calendar_angles: np.ndarray,
    target_inventory: np.ndarray,
    price_normalizer: float,
) -> np.ndarray:
    """Builds polynomial regression features up to degree two.
    
    Args:
        storage: Storage value.
        price: Price value.
        time_fraction: Time fraction value.
        capacity: Capacity value.
        calendar_angles: Calendar angles value.
        target_inventory: Target inventory value.
        price_normalizer: Price normalizer value.
    
    Returns:
        Features result.

    """
    normalized_storage = storage / capacity
    normalized_price = price / max(float(price_normalizer), 1e-8)
    normalized_target = target_inventory / capacity
    time = np.full_like(normalized_storage, time_fraction, dtype=np.float64)
    sin_day = np.sin(calendar_angles)
    cos_day = np.cos(calendar_angles)
    return np.column_stack(
        [
            np.ones_like(normalized_storage),
            normalized_storage,
            normalized_price,
            time,
            normalized_storage**2,
            normalized_price**2,
            normalized_storage * normalized_price,
            normalized_target,
            normalized_target**2,
            normalized_storage * normalized_target,
            normalized_price * normalized_target,
            sin_day,
            cos_day,
            normalized_price * sin_day,
            normalized_price * cos_day,
        ]
    )


def _calendar_angles(start_dates: list[date], step: int) -> np.ndarray:
    """Returns cyclic day-of-year angles for one relative episode step.
    
    Args:
        start_dates: Start dates value.
        step: Step value.
    
    Returns:
        Calendar angles result.

    """
    angles = []
    for start_date in start_dates:
        current_date = start_date + timedelta(days=step)
        days_in_year = 366 if _is_leap_year(current_date.year) else 365
        angles.append(
            2.0 * np.pi * (current_date.timetuple().tm_yday - 1) / days_in_year
        )
    return np.asarray(angles, dtype=np.float64)


def _is_leap_year(year: int) -> bool:
    """Returns whether a Gregorian calendar year is a leap year.
    
    Args:
        year: Year value.
    
    Returns:
        Is leap year result.

    """
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


@dataclass
class LSMCBenchmark:
    """Approximate dynamic-programming benchmark for stochastic valuation."""

    storage_params: StorageParams
    lambda_terminal: float
    action_grid: np.ndarray = field(
        default_factory=lambda: np.array(
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            dtype=np.float64,
        )
    )
    n_inventory_levels: int = 21
    coefficients: dict[int, np.ndarray] = field(default_factory=dict)
    price_normalizer: float = 1.0

    def fit(
        self,
        train_paths: np.ndarray,
        start_dates: list[date] | None = None,
        initial_inventories: np.ndarray | None = None,
        target_inventories: np.ndarray | None = None,
    ) -> "LSMCBenchmark":
        """Fits continuation value regressions on training paths.
        
        Args:
            train_paths: Train paths value.
            start_dates: Start dates value.
            initial_inventories: Initial inventories value.
            target_inventories: Target inventories value.
        
        Returns:
            Fitted model or policy instance.
        
        Raises:
            ValueError: If an input value or configuration is invalid.

        """
        n_paths, horizon = train_paths.shape
        if self.n_inventory_levels < 2:
            raise ValueError("n_inventory_levels must be at least 2")
        if start_dates is None:
            start_dates = [date(2001, 1, 1)] * n_paths
        if len(start_dates) != n_paths:
            raise ValueError("start_dates must contain one date per path")
        initial_values = _inventory_values(
            initial_inventories,
            n_paths,
            self.storage_params.initial_inventory,
        )
        target_values = (
            initial_values
            if target_inventories is None
            else _inventory_values(target_inventories, n_paths, 0.0)
        )
        del initial_values
        self.price_normalizer = float(np.mean(train_paths))
        inventory_grid = np.linspace(
            0.0,
            self.storage_params.capacity,
            int(self.n_inventory_levels),
            dtype=np.float64,
        )
        storage_states = np.tile(inventory_grid, n_paths)
        target_states = np.repeat(target_values, len(inventory_grid))
        for step in reversed(range(horizon)):
            price_states = np.repeat(train_paths[:, step], len(inventory_grid))
            calendar_states = np.repeat(
                _calendar_angles(start_dates, step),
                len(inventory_grid),
            )
            candidates = []
            for action in self.action_grid:
                remaining_steps_after_action = horizon - step - 1
                executed = _clip_actions(
                    action,
                    storage_states,
                    self.storage_params,
                    remaining_steps_after_action,
                    target_states,
                )
                level_next = storage_states + executed
                reward = -executed * price_states
                if step == horizon - 1:
                    value = reward - self.lambda_terminal * np.abs(
                        level_next - target_states
                    )
                else:
                    continuation_features = _features(
                        level_next,
                        price_states,
                        (step + 1) / max(horizon - 1, 1),
                        self.storage_params.capacity,
                        np.repeat(
                            _calendar_angles(start_dates, step + 1),
                            len(inventory_grid),
                        ),
                        target_states,
                        self.price_normalizer,
                    )
                    value = reward + continuation_features @ self.coefficients[step + 1]
                candidates.append(value)
            best_values = np.max(np.vstack(candidates), axis=0)
            x = _features(
                storage_states,
                price_states,
                step / max(horizon - 1, 1),
                self.storage_params.capacity,
                calendar_states,
                target_states,
                self.price_normalizer,
            )
            self.coefficients[step] = np.linalg.lstsq(x, best_values, rcond=None)[0]
        return self

    def predict_action(
        self,
        storage_level: float,
        price: float,
        current_step: int,
        horizon: int,
        start_date: date | None = None,
        target_inventory: float | None = None,
    ) -> float:
        """Chooses the best grid action using fitted continuation estimates.
        
        Args:
            storage_level: Storage level value.
            price: Price value.
            current_step: Current step value.
            horizon: Horizon value.
            start_date: Start date value.
            target_inventory: Target inventory value.
        
        Returns:
            Action selected for the current path state.

        """
        target = (
            self.storage_params.target_terminal_inventory
            if target_inventory is None
            else float(target_inventory)
        )
        best_action = 0.0
        best_value = -np.inf
        remaining_steps_after_action = horizon - current_step - 1
        for action in self.action_grid:
            executed = _clip_action(
                action,
                storage_level,
                self.storage_params,
                remaining_steps_after_action,
                target,
            )
            next_level = storage_level + executed
            immediate = -executed * price
            if current_step == horizon - 1:
                immediate -= self.lambda_terminal * abs(
                    next_level - target
                )
            coef = self.coefficients.get(current_step + 1)
            continuation = 0.0
            if coef is not None and current_step < horizon - 1:
                x = _features(
                    np.array([next_level]),
                    np.array([price]),
                    (current_step + 1) / max(horizon - 1, 1),
                    self.storage_params.capacity,
                    _calendar_angles(
                        [start_date or date(2001, 1, 1)],
                        current_step + 1,
                    ),
                    np.array([target]),
                    self.price_normalizer,
                )
                continuation = float((x @ coef).item())
            value = immediate + continuation
            if value > best_value:
                best_value = value
                best_action = float(executed)
        return best_action

    def evaluate(
        self,
        paths: np.ndarray,
        start_dates: list[date] | None = None,
        initial_inventories: np.ndarray | None = None,
        target_inventories: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Evaluates the fitted LSMC policy on fixed paths.
        
        Args:
            paths: Price paths or filesystem paths to process.
            start_dates: Start dates value.
            initial_inventories: Initial inventories value.
            target_inventories: Target inventories value.
        
        Returns:
            Evaluation metrics dictionary.
        
        Raises:
            ValueError: If an input value or configuration is invalid.

        """
        if start_dates is None:
            start_dates = [date(2001, 1, 1)] * len(paths)
        if len(start_dates) != len(paths):
            raise ValueError("start_dates must contain one date per path")
        initial_values = _inventory_values(
            initial_inventories,
            len(paths),
            self.storage_params.initial_inventory,
        )
        target_values = (
            initial_values
            if target_inventories is None
            else _inventory_values(target_inventories, len(paths), 0.0)
        )
        returns = []
        for path, start_date, initial, target in zip(
            paths,
            start_dates,
            initial_values,
            target_values,
            strict=True,
        ):
            storage = float(initial)
            total = 0.0
            for step, price in enumerate(path):
                action = self.predict_action(
                    storage,
                    float(price),
                    step,
                    len(path),
                    start_date,
                    float(target),
                )
                executed = _clip_action(
                    action,
                    storage,
                    self.storage_params,
                    len(path) - step - 1,
                    float(target),
                )
                storage += executed
                total += -executed * float(price)
                if step == len(path) - 1:
                    total -= self.lambda_terminal * abs(
                        storage - target
                    )
            returns.append(total)
        return {
            "mean_return_raw": float(np.mean(returns)),
            "std_return_raw": float(np.std(returns)),
            "median_return_raw": float(np.median(returns)),
        }


def _clip_action(
    action: float,
    storage_level: float,
    params: StorageParams,
    remaining_steps_after_action: int,
    target_inventory: float,
) -> float:
    """Clips LSMC candidate actions exactly like the environment executes them.
    
    Args:
        action: Action value.
        storage_level: Storage level value.
        params: Params value.
        remaining_steps_after_action: Remaining steps after action value.
        target_inventory: Target inventory value.
    
    Returns:
        Clip action result.

    """
    return _scalar_clip_action(
        action,
        storage_level,
        params,
        remaining_steps_after_action,
        target_inventory,
    )


def _clip_actions(
    action: float,
    storage_levels: np.ndarray,
    params: StorageParams,
    remaining_steps_after_action: int,
    target_inventories: np.ndarray,
) -> np.ndarray:
    """Vectorizes environment-equivalent action clipping for LSMC candidates.
    
    Args:
        action: Action value.
        storage_levels: Storage levels value.
        params: Params value.
        remaining_steps_after_action: Remaining steps after action value.
        target_inventories: Target inventories value.
    
    Returns:
        Clip actions result.

    """
    storage = np.asarray(storage_levels, dtype=np.float64)
    target = np.asarray(target_inventories, dtype=np.float64)
    normal_lower = np.maximum(-params.withdrawal_rate, -storage)
    normal_upper = np.minimum(params.injection_rate, params.capacity - storage)

    remaining_steps = max(int(remaining_steps_after_action), 0)
    min_reachable = np.maximum(
        0.0,
        target - remaining_steps * params.injection_rate,
    )
    max_reachable = np.minimum(
        params.capacity,
        target + remaining_steps * params.withdrawal_rate,
    )
    terminal_lower = min_reachable - storage
    terminal_upper = max_reachable - storage
    lower = np.maximum(normal_lower, terminal_lower)
    upper = np.minimum(normal_upper, terminal_upper)

    requested = np.full_like(storage, float(action), dtype=np.float64)
    normal_clipped = np.minimum(np.maximum(requested, normal_lower), normal_upper)
    terminal_clipped = np.minimum(np.maximum(requested, lower), upper)
    return np.where(lower > upper, normal_clipped, terminal_clipped)


def _scalar_clip_action(
    action: float,
    storage_level: float,
    params: StorageParams,
    remaining_steps_after_action: int,
    target_inventory: float,
) -> float:
    """Returns scalar environment clipping for tests and implementation parity.
    
    Args:
        action: Action value.
        storage_level: Storage level value.
        params: Params value.
        remaining_steps_after_action: Remaining steps after action value.
        target_inventory: Target inventory value.
    
    Returns:
        Scalar clip action result.

    """
    return clip_storage_action_to_terminal_feasibility(
        requested_action=float(action),
        storage_level=float(storage_level),
        params=params,
        remaining_steps_after_action=remaining_steps_after_action,
        target_inventory=float(target_inventory),
    )


def _inventory_values(
    values: np.ndarray | None,
    n_paths: int,
    default: float,
) -> np.ndarray:
    """Returns a validated inventory vector for path-wise LSMC state.
    
    Args:
        values: Numeric values to transform or summarize.
        n_paths: Number of price paths to generate or evaluate.
        default: Default value.
    
    Returns:
        Inventory values result.
    
    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    if values is None:
        return np.full(n_paths, default, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n_paths,):
        raise ValueError("Inventory values must contain one value per path")
    return array
