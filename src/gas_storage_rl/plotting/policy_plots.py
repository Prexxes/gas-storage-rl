"""Policy and trajectory diagnostic plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _matrix(trajectories: list[dict], key: str, n_paths: int) -> np.ndarray:
    """Extracts a trajectory matrix for an info key."""
    return np.array(
        [[info[key] for info in trajectory["infos"]] for trajectory in trajectories[:n_paths]]
    )


def plot_agent_actions(trajectories: list[dict], n_paths: int = 50, include_mean: bool = True):
    """Plots executed action paths."""
    return _line_plot(trajectories, "executed_action", "Executed action", n_paths, include_mean)


def plot_storage_levels(trajectories: list[dict], n_paths: int = 50, include_mean: bool = True):
    """Plots storage level paths."""
    return _line_plot(trajectories, "storage_level", "Storage level", n_paths, include_mean)


def plot_cumulative_cashflows(trajectories: list[dict], n_paths: int = 50, include_mean: bool = True):
    """Plots cumulative raw cashflow paths."""
    cashflows = _matrix(trajectories, "raw_cashflow", n_paths)
    cumulative = np.cumsum(cashflows, axis=1)
    figure, axis = plt.subplots()
    axis.plot(cumulative.T, color="tab:green", alpha=0.15, linewidth=0.8)
    if include_mean:
        axis.plot(np.mean(cumulative, axis=0), color="black", linewidth=2.0)
    axis.set_xlabel("Timesteps")
    axis.set_ylabel("Cumulative cashflow")
    return figure


def plot_price_action_scatter(trajectories: list[dict]):
    """Plots gas price against executed action."""
    prices = _matrix(trajectories, "price", len(trajectories)).reshape(-1)
    actions = _matrix(trajectories, "executed_action", len(trajectories)).reshape(-1)
    figure, axis = plt.subplots()
    axis.scatter(prices, actions, alpha=0.2, s=8)
    axis.set_xlabel("Gas spot price")
    axis.set_ylabel("Executed action")
    return figure


def _line_plot(
    trajectories: list[dict],
    key: str,
    ylabel: str,
    n_paths: int,
    include_mean: bool,
):
    """Creates a reusable trajectory line plot."""
    values = _matrix(trajectories, key, n_paths)
    figure, axis = plt.subplots()
    axis.plot(values.T, color="tab:orange", alpha=0.15, linewidth=0.8)
    if include_mean:
        axis.plot(np.mean(values, axis=0), color="black", linewidth=2.0)
    axis.set_xlabel("Timesteps")
    axis.set_ylabel(ylabel)
    return figure
