"""Least-Squares Monte Carlo benchmark with a discrete action grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from gas_storage_rl.envs.storage_dynamics import StorageParams, clip_storage_action


def _features(
    storage: np.ndarray,
    price: np.ndarray,
    time_fraction: float,
    capacity: float,
    calendar_angles: np.ndarray,
    target_inventory: np.ndarray,
) -> np.ndarray:
    """Builds polynomial regression features up to degree two."""
    normalized_storage = storage / capacity
    normalized_price = price / np.maximum(np.mean(price), 1e-8)
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
    """Returns cyclic day-of-year angles for one relative episode step."""
    angles = []
    for start_date in start_dates:
        current_date = start_date + timedelta(days=step)
        days_in_year = 366 if _is_leap_year(current_date.year) else 365
        angles.append(
            2.0 * np.pi * (current_date.timetuple().tm_yday - 1) / days_in_year
        )
    return np.asarray(angles, dtype=np.float64)


def _is_leap_year(year: int) -> bool:
    """Returns whether a Gregorian calendar year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


@dataclass
class LSMCBenchmark:
    """Approximate dynamic-programming benchmark for stochastic valuation."""

    storage_params: StorageParams
    lambda_terminal: float
    action_grid: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    )
    coefficients: dict[int, np.ndarray] = field(default_factory=dict)

    def fit(
        self,
        train_paths: np.ndarray,
        start_dates: list[date] | None = None,
        initial_inventories: np.ndarray | None = None,
        target_inventories: np.ndarray | None = None,
    ) -> "LSMCBenchmark":
        """Fits continuation value regressions on training paths."""
        n_paths, horizon = train_paths.shape
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
        storage = initial_values.copy()
        # A simple rollout state proxy keeps this MVP small and reproducible.
        values_next = np.zeros(n_paths, dtype=np.float64)
        for step in reversed(range(horizon)):
            price = train_paths[:, step]
            candidates = []
            next_levels = []
            for action in self.action_grid:
                executed = np.array(
                    [
                        clip_storage_action(action, level, self.storage_params)
                        for level in storage
                    ]
                )
                level_next = storage + executed
                reward = -executed * price
                if step == horizon - 1:
                    reward -= self.lambda_terminal * np.abs(
                        level_next - target_values
                    )
                candidates.append(reward + values_next)
                next_levels.append(level_next)
            best_values = np.max(np.vstack(candidates), axis=0)
            x = _features(
                storage,
                price,
                step / max(horizon - 1, 1),
                self.storage_params.capacity,
                _calendar_angles(start_dates, step),
                target_values,
            )
            self.coefficients[step] = np.linalg.lstsq(x, best_values, rcond=None)[0]
            best_action_index = np.argmax(np.vstack(candidates), axis=0)
            storage = np.array(
                [next_levels[index][path] for path, index in enumerate(best_action_index)]
            )
            values_next = best_values
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
        """Chooses the best grid action using fitted continuation estimates."""
        target = (
            self.storage_params.target_terminal_inventory
            if target_inventory is None
            else float(target_inventory)
        )
        best_action = 0.0
        best_value = -np.inf
        for action in self.action_grid:
            executed = clip_storage_action(action, storage_level, self.storage_params)
            next_level = storage_level + executed
            immediate = -executed * price
            if current_step == horizon - 1:
                immediate -= self.lambda_terminal * abs(
                    next_level - target
                )
            coef = self.coefficients.get(current_step)
            continuation = 0.0
            if coef is not None and current_step < horizon - 1:
                x = _features(
                    np.array([next_level]),
                    np.array([price]),
                    current_step / max(horizon - 1, 1),
                    self.storage_params.capacity,
                    _calendar_angles(
                        [start_date or date(2001, 1, 1)],
                        current_step,
                    ),
                    np.array([target]),
                )
                continuation = float((x @ coef).item())
            value = immediate + continuation
            if value > best_value:
                best_value = value
                best_action = float(action)
        return best_action

    def evaluate(
        self,
        paths: np.ndarray,
        start_dates: list[date] | None = None,
        initial_inventories: np.ndarray | None = None,
        target_inventories: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Evaluates the fitted LSMC policy on fixed paths."""
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
                executed = clip_storage_action(action, storage, self.storage_params)
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


def _inventory_values(
    values: np.ndarray | None,
    n_paths: int,
    default: float,
) -> np.ndarray:
    """Returns a validated inventory vector for path-wise LSMC state."""
    if values is None:
        return np.full(n_paths, default, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n_paths,):
        raise ValueError("Inventory values must contain one value per path")
    return array
