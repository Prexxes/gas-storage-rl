"""Least-Squares Monte Carlo benchmark with a discrete action grid."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gas_storage_rl.envs.storage_dynamics import StorageParams, clip_storage_action


def _features(storage: np.ndarray, price: np.ndarray, time_fraction: float, capacity: float):
    """Builds polynomial regression features up to degree two."""
    normalized_storage = storage / capacity
    normalized_price = price / np.maximum(np.mean(price), 1e-8)
    time = np.full_like(normalized_storage, time_fraction, dtype=np.float64)
    return np.column_stack(
        [
            np.ones_like(normalized_storage),
            normalized_storage,
            normalized_price,
            time,
            normalized_storage**2,
            normalized_price**2,
            normalized_storage * normalized_price,
        ]
    )


@dataclass
class LSMCBenchmark:
    """Approximate dynamic-programming benchmark for stochastic valuation."""

    storage_params: StorageParams
    lambda_terminal: float
    action_grid: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    )
    coefficients: dict[int, np.ndarray] = field(default_factory=dict)

    def fit(self, train_paths: np.ndarray) -> "LSMCBenchmark":
        """Fits continuation value regressions on training paths."""
        n_paths, horizon = train_paths.shape
        storage = np.zeros(n_paths, dtype=np.float64)
        # A simple rollout state proxy keeps this MVP small and reproducible.
        values_next = np.zeros(n_paths, dtype=np.float64)
        terminal_target = self.storage_params.target_terminal_inventory
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
                        level_next - terminal_target
                    )
                candidates.append(reward + values_next)
                next_levels.append(level_next)
            best_values = np.max(np.vstack(candidates), axis=0)
            x = _features(storage, price, step / max(horizon - 1, 1), self.storage_params.capacity)
            self.coefficients[step] = np.linalg.lstsq(x, best_values, rcond=None)[0]
            best_action_index = np.argmax(np.vstack(candidates), axis=0)
            storage = np.array(
                [next_levels[index][path] for path, index in enumerate(best_action_index)]
            )
            values_next = best_values
        return self

    def predict_action(self, storage_level: float, price: float, current_step: int, horizon: int) -> float:
        """Chooses the best grid action using fitted continuation estimates."""
        best_action = 0.0
        best_value = -np.inf
        for action in self.action_grid:
            executed = clip_storage_action(action, storage_level, self.storage_params)
            next_level = storage_level + executed
            immediate = -executed * price
            if current_step == horizon - 1:
                immediate -= self.lambda_terminal * abs(
                    next_level - self.storage_params.target_terminal_inventory
                )
            coef = self.coefficients.get(current_step)
            continuation = 0.0
            if coef is not None and current_step < horizon - 1:
                x = _features(
                    np.array([next_level]),
                    np.array([price]),
                    current_step / max(horizon - 1, 1),
                    self.storage_params.capacity,
                )
                continuation = float((x @ coef).item())
            value = immediate + continuation
            if value > best_value:
                best_value = value
                best_action = float(action)
        return best_action

    def evaluate(self, paths: np.ndarray) -> dict[str, float]:
        """Evaluates the fitted LSMC policy on fixed paths."""
        returns = []
        for path in paths:
            storage = self.storage_params.initial_inventory
            total = 0.0
            for step, price in enumerate(path):
                action = self.predict_action(storage, float(price), step, len(path))
                executed = clip_storage_action(action, storage, self.storage_params)
                storage += executed
                total += -executed * float(price)
                if step == len(path) - 1:
                    total -= self.lambda_terminal * abs(
                        storage - self.storage_params.target_terminal_inventory
                    )
            returns.append(total)
        return {
            "mean_return_raw": float(np.mean(returns)),
            "std_return_raw": float(np.std(returns)),
            "median_return_raw": float(np.median(returns)),
        }
