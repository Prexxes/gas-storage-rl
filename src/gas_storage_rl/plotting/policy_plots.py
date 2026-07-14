"""Policy and trajectory diagnostic plots."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

TimeAxis = Literal["episode", "absolute", "day_of_year"]


def _matrix(trajectories: list[dict], key: str, n_paths: int) -> np.ndarray:
    """Extracts a trajectory matrix for an info key.
    
    Args:
        trajectories: Rollout trajectories to summarize or transform.
        key: Key value.
        n_paths: Number of price paths to generate or evaluate.
    
    Returns:
        Matrix result.

    """
    return np.array(
        [[info[key] for info in trajectory["infos"]] for trajectory in trajectories[:n_paths]]
    )


def plot_agent_actions(
    trajectories: list[dict],
    n_paths: int = 50,
    include_mean: bool = True,
    time_axis: TimeAxis = "episode",
):
    """Plots executed action paths.
    
    Args:
        trajectories: Rollout trajectories to summarize or transform.
        n_paths: Number of price paths to generate or evaluate.
        include_mean: Include mean value.
        time_axis: Time axis value.
    
    Returns:
        Computed result.

    """
    return _line_plot(
        trajectories,
        "executed_action",
        "Executed action",
        n_paths,
        include_mean,
        time_axis,
    )


def plot_storage_levels(
    trajectories: list[dict],
    n_paths: int = 50,
    include_mean: bool = True,
    time_axis: TimeAxis = "episode",
):
    """Plots storage level paths.
    
    Args:
        trajectories: Rollout trajectories to summarize or transform.
        n_paths: Number of price paths to generate or evaluate.
        include_mean: Include mean value.
        time_axis: Time axis value.
    
    Returns:
        Computed result.

    """
    return _line_plot(
        trajectories,
        "storage_level",
        "Storage level",
        n_paths,
        include_mean,
        time_axis,
    )


def plot_cumulative_cashflows(
    trajectories: list[dict],
    n_paths: int = 50,
    include_mean: bool = True,
    time_axis: TimeAxis = "episode",
):
    """Plots cumulative raw cashflow paths.
    
    Args:
        trajectories: Rollout trajectories to summarize or transform.
        n_paths: Number of price paths to generate or evaluate.
        include_mean: Include mean value.
        time_axis: Time axis value.
    
    Returns:
        Computed result.

    """
    figure, axis = plt.subplots()
    aggregate_values: dict[int, list[float]] = defaultdict(list)
    for trajectory in trajectories[:n_paths]:
        x_values = _time_values(trajectory["infos"], time_axis)
        cashflows = [info["raw_cashflow"] for info in trajectory["infos"]]
        cumulative = np.cumsum(cashflows)
        _plot_values(axis, x_values, cumulative, "tab:green", time_axis)
        for x_value, value in zip(x_values, cumulative, strict=True):
            aggregate_values[int(x_value)].append(float(value))
    if include_mean:
        _plot_aggregate(axis, aggregate_values, time_axis)
    axis.set_xlabel(_time_axis_label(time_axis))
    axis.set_ylabel("Cumulative cashflow")
    return figure


def plot_price_action_scatter(trajectories: list[dict]):
    """Plots gas price against executed action.
    
    Args:
        trajectories: Rollout trajectories to summarize or transform.
    
    Returns:
        Computed result.

    """
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
    time_axis: TimeAxis,
):
    """Creates a reusable trajectory line plot.
    
    Args:
        trajectories: Rollout trajectories to summarize or transform.
        key: Key value.
        ylabel: Ylabel value.
        n_paths: Number of price paths to generate or evaluate.
        include_mean: Include mean value.
        time_axis: Time axis value.
    
    Returns:
        Line plot result.

    """
    figure, axis = plt.subplots()
    aggregate_values: dict[int, list[float]] = defaultdict(list)
    for trajectory in trajectories[:n_paths]:
        x_values = _time_values(trajectory["infos"], time_axis)
        y_values = [info[key] for info in trajectory["infos"]]
        _plot_values(axis, x_values, y_values, "tab:orange", time_axis)
        for x_value, value in zip(x_values, y_values, strict=True):
            aggregate_values[int(x_value)].append(float(value))
    if include_mean:
        _plot_aggregate(axis, aggregate_values, time_axis)
    axis.set_xlabel(_time_axis_label(time_axis))
    axis.set_ylabel(ylabel)
    return figure


def _time_values(infos: list[dict], time_axis: TimeAxis) -> np.ndarray:
    """Returns x-axis values for trajectory infos.
    
    Args:
        infos: Per-step environment information dictionaries.
        time_axis: Time axis value.
    
    Returns:
        Time values result.
    
    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    if time_axis == "episode":
        return np.array([info["current_step"] for info in infos], dtype=np.int64)
    if time_axis == "absolute":
        return np.array(
            [info["start_index"] + info["current_step"] for info in infos],
            dtype=np.int64,
        )
    if time_axis == "day_of_year":
        return np.array(
            [
                (info["start_index"] + info["current_step"]) % 365 + 1
                for info in infos
            ],
            dtype=np.int64,
        )
    raise ValueError(f"Unsupported time_axis: {time_axis}")


def _time_axis_label(time_axis: TimeAxis) -> str:
    """Returns a human-readable x-axis label.
    
    Args:
        time_axis: Time axis value.
    
    Returns:
        Time axis label result.
    
    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    if time_axis == "episode":
        return "Episode timestep"
    if time_axis == "absolute":
        return "Simulation day"
    if time_axis == "day_of_year":
        return "Day of year"
    raise ValueError(f"Unsupported time_axis: {time_axis}")


def _plot_values(
    axis: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray | list[float],
    color: str,
    time_axis: TimeAxis,
) -> None:
    """Plots individual trajectory values for the selected time axis.
    
    Args:
        axis: Axis value.
        x_values: X values value.
        y_values: Y values value.
        color: Color value.
        time_axis: Time axis value.

    """
    if time_axis == "day_of_year":
        axis.scatter(x_values, y_values, color=color, alpha=0.15, s=8)
        axis.set_xlim(1, 365)
        return
    axis.plot(x_values, y_values, color=color, alpha=0.15, linewidth=0.8)


def _plot_aggregate(
    axis: plt.Axes,
    aggregate_values: dict[int, list[float]],
    time_axis: TimeAxis,
) -> None:
    """Plots mean or seasonal quantile aggregates.
    
    Args:
        axis: Axis value.
        aggregate_values: Aggregate values value.
        time_axis: Time axis value.

    """
    x_values = np.array(sorted(aggregate_values), dtype=np.int64)
    if len(x_values) == 0:
        return
    if time_axis == "day_of_year":
        medians = np.array(
            [np.median(aggregate_values[int(x_value)]) for x_value in x_values]
        )
        p10 = np.array(
            [np.percentile(aggregate_values[int(x_value)], 10) for x_value in x_values]
        )
        p90 = np.array(
            [np.percentile(aggregate_values[int(x_value)], 90) for x_value in x_values]
        )
        axis.fill_between(x_values, p10, p90, color="black", alpha=0.12, linewidth=0)
        axis.plot(x_values, medians, color="black", linewidth=2.0)
        axis.set_xlim(1, 365)
        return
    means = np.array(
        [np.mean(aggregate_values[int(x_value)]) for x_value in x_values]
    )
    axis.plot(x_values, means, color="black", linewidth=2.0)
