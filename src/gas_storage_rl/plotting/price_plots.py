"""Price path plots."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

TimeAxis = Literal["episode", "absolute", "day_of_year"]


def plot_price_paths(
    paths: np.ndarray,
    n_paths: int = 50,
    include_mean: bool = True,
    start_indices: np.ndarray | None = None,
    time_axis: TimeAxis = "episode",
):
    """Plots individual and mean gas spot price paths."""
    figure, axis = plt.subplots()
    selected = paths[:n_paths]
    starts = _start_indices(start_indices, len(selected))
    aggregate_values: dict[int, list[float]] = defaultdict(list)
    for path, start_index in zip(selected, starts, strict=True):
        x_values = _time_values(len(path), int(start_index), time_axis)
        if time_axis == "day_of_year":
            axis.scatter(x_values, path, color="tab:blue", alpha=0.15, s=8)
            axis.set_xlim(1, 365)
        else:
            axis.plot(x_values, path, color="tab:blue", alpha=0.15, linewidth=0.8)
        for x_value, value in zip(x_values, path, strict=True):
            aggregate_values[int(x_value)].append(float(value))
    if include_mean:
        _plot_aggregate(axis, aggregate_values, time_axis)
    axis.set_xlabel(_time_axis_label(time_axis))
    axis.set_ylabel("Gas spot price")
    return figure


def _start_indices(start_indices: np.ndarray | None, n_paths: int) -> np.ndarray:
    """Returns start indices for selected paths."""
    if start_indices is None:
        return np.zeros(n_paths, dtype=np.int64)
    return np.asarray(start_indices[:n_paths], dtype=np.int64)


def _time_values(path_length: int, start_index: int, time_axis: TimeAxis) -> np.ndarray:
    """Returns x-axis values for a path."""
    steps = np.arange(path_length, dtype=np.int64)
    if time_axis == "episode":
        return steps
    if time_axis == "absolute":
        return start_index + steps
    if time_axis == "day_of_year":
        return (start_index + steps) % 365 + 1
    raise ValueError(f"Unsupported time_axis: {time_axis}")


def _time_axis_label(time_axis: TimeAxis) -> str:
    """Returns a human-readable x-axis label."""
    if time_axis == "episode":
        return "Episode timestep"
    if time_axis == "absolute":
        return "Simulation day"
    if time_axis == "day_of_year":
        return "Day of year"
    raise ValueError(f"Unsupported time_axis: {time_axis}")


def _plot_aggregate(
    axis: plt.Axes,
    aggregate_values: dict[int, list[float]],
    time_axis: TimeAxis,
) -> None:
    """Plots mean or seasonal quantile aggregates."""
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
