"""Perfect-foresight linear-programming upper bound."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog

from gas_storage_rl.envs.storage_dynamics import StorageParams


@dataclass
class PerfectForesightResult:
    """Solution container for one perfect-foresight path."""

    objective_value: float
    actions: np.ndarray
    storage_levels: np.ndarray
    terminal_deviation: float
    success: bool


class PerfectForesightBaseline:
    """Solves each known price path with full future information."""

    def __init__(self, storage_params: StorageParams, lambda_terminal: float) -> None:
        """Initializes the baseline."""
        self.storage_params = storage_params
        self.lambda_terminal = float(lambda_terminal)

    def solve_path(self, prices: np.ndarray) -> PerfectForesightResult:
        """Solves the finite-horizon linear program for one path."""
        horizon = len(prices)
        n_actions = horizon
        n_levels = horizon + 1
        d_index = n_actions + n_levels
        n_variables = d_index + 1

        c = np.zeros(n_variables)
        c[:n_actions] = prices
        c[d_index] = self.lambda_terminal

        bounds = []
        bounds.extend(
            [(-self.storage_params.withdrawal_rate, self.storage_params.injection_rate)]
            * n_actions
        )
        bounds.extend([(0.0, self.storage_params.capacity)] * n_levels)
        bounds.append((0.0, None))

        a_eq = []
        b_eq = []
        row = np.zeros(n_variables)
        row[n_actions] = 1.0
        a_eq.append(row)
        b_eq.append(self.storage_params.initial_inventory)
        for step in range(horizon):
            row = np.zeros(n_variables)
            row[n_actions + step + 1] = 1.0
            row[n_actions + step] = -1.0
            row[step] = -1.0
            a_eq.append(row)
            b_eq.append(0.0)

        a_ub = []
        b_ub = []
        target = self.storage_params.target_terminal_inventory
        row = np.zeros(n_variables)
        row[n_actions + horizon] = 1.0
        row[d_index] = -1.0
        a_ub.append(row)
        b_ub.append(target)
        row = np.zeros(n_variables)
        row[n_actions + horizon] = -1.0
        row[d_index] = -1.0
        a_ub.append(row)
        b_ub.append(-target)

        result = linprog(
            c,
            A_ub=np.array(a_ub),
            b_ub=np.array(b_ub),
            A_eq=np.array(a_eq),
            b_eq=np.array(b_eq),
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return PerfectForesightResult(
                objective_value=float("nan"),
                actions=np.zeros(horizon),
                storage_levels=np.zeros(horizon + 1),
                terminal_deviation=float("nan"),
                success=False,
            )
        actions = result.x[:n_actions]
        storage_levels = result.x[n_actions : n_actions + n_levels]
        terminal_deviation = result.x[d_index]
        return PerfectForesightResult(
            objective_value=float(-result.fun),
            actions=actions,
            storage_levels=storage_levels,
            terminal_deviation=float(terminal_deviation),
            success=True,
        )

    def evaluate_paths(self, paths: np.ndarray) -> dict[str, float]:
        """Evaluates the upper bound over multiple known paths."""
        values = [self.solve_path(path).objective_value for path in paths]
        return {
            "mean_return_raw": float(np.mean(values)),
            "median_return_raw": float(np.median(values)),
            "std_return_raw": float(np.std(values)),
        }

    def solve_paths(
        self,
        paths: np.ndarray,
        split: str,
        date_ranges: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Solves and serializes perfect-foresight trajectories for fixed paths."""
        trajectories = []
        for path_id, prices in enumerate(paths):
            result = self.solve_path(prices)
            date_range = date_ranges[path_id] if date_ranges is not None else {}
            trajectories.append(
                {
                    "split": split,
                    "path_id": path_id,
                    "start_date": date_range.get("start_date"),
                    "end_date": date_range.get("end_date"),
                    "prices": prices.astype(float).tolist(),
                    "actions": result.actions.astype(float).tolist(),
                    "storage_levels": result.storage_levels.astype(float).tolist(),
                    "objective_value": result.objective_value,
                    "terminal_deviation": result.terminal_deviation,
                    "success": result.success,
                }
            )
        return trajectories
