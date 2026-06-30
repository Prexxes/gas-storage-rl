"""Diagnostics for perfect-foresight trajectories used in pretraining."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gas_storage_rl.pretraining.behavior_cloning import load_trajectory_rows

ACTION_COLUMNS = ["-1", "0", "+1"]
ACTION_COLORS = {"-1": "#377eb8", "0": "#999999", "+1": "#e41a1c"}


def classify_action(action: float, tolerance: float = 1e-6) -> str:
    """Maps a continuous storage action to withdrawal, hold, or injection."""
    if action > tolerance:
        return "+1"
    if action < -tolerance:
        return "-1"
    return "0"


def default_output_dir(trajectory_path: str | Path) -> Path:
    """Returns a diagnostics output directory derived from the trajectory path."""
    path = Path(trajectory_path)
    if path.parent.parent.name == "benchmarks":
        return path.parent.parent.parent / "diagnostics" / path.parent.name
    return path.parent / "diagnostics"


def trajectory_rows_to_step_samples(
    trajectories: list[dict[str, Any]],
    action_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Converts path-level perfect-foresight rows to one row per decision step.

    The ``absolute_day`` column is the global simulation-day index within the raw
    price path, where day 0 is January 1 of the first synthetic year. It is computed
    as ``start_index + episode_day``. Older trajectory files without ``start_index``
    are supported when they include ``start_date``.
    """
    rows = []
    for trajectory_index, trajectory in enumerate(trajectories):
        prices = np.asarray(trajectory["prices"], dtype=np.float64)
        actions = np.asarray(trajectory["actions"], dtype=np.float64)
        storage_levels = np.asarray(trajectory["storage_levels"], dtype=np.float64)
        if len(prices) != len(actions):
            raise ValueError("prices and actions must have the same length")
        if len(storage_levels) < len(actions):
            raise ValueError("storage_levels must include each pre-action level")
        horizon = len(actions)
        denominator = max(horizon - 1, 1)
        path_id = int(trajectory.get("path_id", trajectory_index))
        target_inventory = float(
            trajectory.get("target_inventory", storage_levels[0])
        )
        start_index = _trajectory_start_index(trajectory)
        for step, action in enumerate(actions):
            rows.append(
                {
                    "split": str(trajectory.get("split", "")),
                    "path_id": path_id,
                    "episode_day": step,
                    "start_index": start_index,
                    "absolute_day": start_index + step,
                    "price": float(prices[step]),
                    "storage_level": float(storage_levels[step]),
                    "remaining_time": float((horizon - 1 - step) / denominator),
                    "target_inventory": target_inventory,
                    "action": float(action),
                    "action_class": classify_action(
                        float(action), tolerance=action_tolerance
                    ),
                }
            )
    if not rows:
        raise ValueError("No trajectory samples found")
    return pd.DataFrame(rows)


def _trajectory_start_index(trajectory: dict[str, Any]) -> int:
    """Returns the raw-path start index for a serialized trajectory."""
    if trajectory.get("start_index") is not None:
        return int(trajectory["start_index"])
    start_date_value = trajectory.get("start_date")
    if start_date_value:
        start_date = datetime.strptime(str(start_date_value), "%Y-%m-%d").date()
        year_start = start_date.replace(month=1, day=1)
        return (start_date - year_start).days
    return 0


def action_counts_by_day(samples: pd.DataFrame) -> pd.DataFrame:
    """Calculates action counts and ambiguity metrics for each absolute day."""
    counts = (
        samples.groupby(["absolute_day", "action_class"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for action_class in ACTION_COLUMNS:
        if action_class not in counts:
            counts[action_class] = 0
    counts = counts[ACTION_COLUMNS].sort_index()
    counts["n_samples"] = counts[ACTION_COLUMNS].sum(axis=1)
    probabilities = counts[ACTION_COLUMNS].div(counts["n_samples"], axis=0)
    counts["majority_action"] = probabilities.idxmax(axis=1)
    counts["majority_share"] = probabilities.max(axis=1)
    counts["n_distinct_actions"] = (counts[ACTION_COLUMNS] > 0).sum(axis=1)
    positive_probabilities = probabilities.where(probabilities > 0.0)
    counts["entropy"] = -(
        positive_probabilities * np.log2(positive_probabilities)
    ).sum(axis=1)
    counts["gini"] = 1.0 - (probabilities**2).sum(axis=1)
    return counts.reset_index()


def feature_stats_by_day_action(samples: pd.DataFrame) -> pd.DataFrame:
    """Summarizes observable state features by absolute day and action class."""
    return (
        samples.groupby(["absolute_day", "action_class"], observed=True)
        .agg(
            n=("action_class", "size"),
            price_mean=("price", "mean"),
            price_std=("price", "std"),
            price_q10=("price", lambda values: values.quantile(0.10)),
            price_q50=("price", "median"),
            price_q90=("price", lambda values: values.quantile(0.90)),
            storage_mean=("storage_level", "mean"),
            storage_std=("storage_level", "std"),
            storage_q10=("storage_level", lambda values: values.quantile(0.10)),
            storage_q50=("storage_level", "median"),
            storage_q90=("storage_level", lambda values: values.quantile(0.90)),
            remaining_time_mean=("remaining_time", "mean"),
            remaining_time_std=("remaining_time", "std"),
            target_inventory_mean=("target_inventory", "mean"),
            target_inventory_std=("target_inventory", "std"),
        )
        .reset_index()
        .sort_values(["absolute_day", "action_class"])
    )


def plot_action_distribution_by_day(
    counts: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Writes a stacked bar chart with action shares by absolute day."""
    indexed = counts.set_index("absolute_day")
    action_counts = indexed[ACTION_COLUMNS]
    action_shares = action_counts.div(action_counts.sum(axis=1), axis=0)
    figure_width = max(8.0, min(24.0, 0.18 * len(action_shares)))
    figure, axis = plt.subplots(figsize=(figure_width, 5.0))
    action_shares.plot(
        kind="bar",
        stacked=True,
        ax=axis,
        color=[ACTION_COLORS[column] for column in ACTION_COLUMNS],
        width=1.0,
    )
    axis.set_xlabel("Absolute day")
    axis.set_ylabel("Action share")
    axis.set_ylim(0.0, 1.0)
    axis.legend(title="Action")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_selected_day_diagnostics(
    samples: pd.DataFrame,
    absolute_day: int,
    output_dir: str | Path,
) -> None:
    """Writes action, price, storage, and price-storage plots for one day."""
    day_samples = samples[samples["absolute_day"] == absolute_day]
    if day_samples.empty:
        raise ValueError(f"No samples found for absolute day {absolute_day}")
    output_path = Path(output_dir)
    _plot_selected_day_action_counts(day_samples, absolute_day, output_path)
    _plot_selected_day_boxplot(
        day_samples,
        absolute_day,
        output_path / f"day_{absolute_day:03d}_price_by_action.png",
        "price",
        "Price",
    )
    _plot_selected_day_boxplot(
        day_samples,
        absolute_day,
        output_path / f"day_{absolute_day:03d}_storage_by_action.png",
        "storage_level",
        "Storage level",
    )
    _plot_selected_day_price_storage(day_samples, absolute_day, output_path)


def run_analysis(
    trajectory_path: str | Path,
    output_dir: str | Path | None = None,
    selected_days: list[int] | None = None,
    action_tolerance: float = 1e-6,
) -> Path:
    """Runs trajectory diagnostics and returns the output directory."""
    output_path = Path(output_dir) if output_dir is not None else default_output_dir(
        trajectory_path
    )
    output_path.mkdir(parents=True, exist_ok=True)
    trajectories = load_trajectory_rows(trajectory_path)
    samples = trajectory_rows_to_step_samples(
        trajectories,
        action_tolerance=action_tolerance,
    )
    counts = action_counts_by_day(samples)
    ambiguous_days = counts.sort_values(
        ["entropy", "majority_share", "absolute_day"],
        ascending=[False, True, True],
    )
    feature_stats = feature_stats_by_day_action(samples)

    samples.to_csv(output_path / "step_samples.csv", index=False)
    counts.to_csv(output_path / "action_counts_by_day.csv", index=False)
    ambiguous_days.to_csv(output_path / "ambiguous_days.csv", index=False)
    feature_stats.to_csv(output_path / "feature_stats_by_day_action.csv", index=False)
    plot_action_distribution_by_day(
        counts,
        output_path / "action_distribution_by_day.png",
    )

    if selected_days:
        selected = samples[samples["absolute_day"].isin(selected_days)]
        selected_stats = feature_stats_by_day_action(selected)
        selected_stats.to_csv(
            output_path / "feature_stats_selected_days.csv",
            index=False,
        )
        for day in selected_days:
            plot_selected_day_diagnostics(samples, day, output_path)

    return output_path


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze perfect-foresight pretraining trajectories.",
    )
    parser.add_argument(
        "--trajectories",
        required=True,
        type=Path,
        help="Path to perfect_foresight_trajectories_<split>.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for CSV and PNG outputs. Defaults to "
            "runs/diagnostics/<run_id> for benchmark trajectory files."
        ),
    )
    parser.add_argument(
        "--day",
        action="append",
        type=int,
        default=[],
        help="Relative absolute day to inspect in detail. Can be passed repeatedly.",
    )
    parser.add_argument(
        "--action-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance around zero for assigning actions to the 0 class.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the command line interface."""
    args = parse_args()
    output_dir = run_analysis(
        args.trajectories,
        output_dir=args.output_dir,
        selected_days=args.day,
        action_tolerance=args.action_tolerance,
    )
    print(f"Wrote diagnostics to {output_dir}")


def _plot_selected_day_action_counts(
    day_samples: pd.DataFrame,
    absolute_day: int,
    output_dir: Path,
) -> None:
    """Writes an action-count plot for one selected absolute day."""
    counts = day_samples["action_class"].value_counts().reindex(
        ACTION_COLUMNS,
        fill_value=0,
    )
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    axis.bar(
        ACTION_COLUMNS,
        [counts[action_class] for action_class in ACTION_COLUMNS],
        color=[ACTION_COLORS[action_class] for action_class in ACTION_COLUMNS],
    )
    axis.set_xlabel("Action")
    axis.set_ylabel("Count")
    axis.set_title(f"Absolute day {absolute_day}")
    figure.tight_layout()
    figure.savefig(output_dir / f"day_{absolute_day:03d}_action_counts.png", dpi=160)
    plt.close(figure)


def _plot_selected_day_boxplot(
    day_samples: pd.DataFrame,
    absolute_day: int,
    output_path: Path,
    column: str,
    ylabel: str,
) -> None:
    """Writes a feature boxplot grouped by action class."""
    data = [
        day_samples.loc[day_samples["action_class"] == action_class, column].to_numpy()
        for action_class in ACTION_COLUMNS
    ]
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    axis.boxplot(data)
    axis.set_xticks(range(1, len(ACTION_COLUMNS) + 1), ACTION_COLUMNS)
    axis.set_xlabel("Action")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Absolute day {absolute_day}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_selected_day_price_storage(
    day_samples: pd.DataFrame,
    absolute_day: int,
    output_dir: Path,
) -> None:
    """Writes a price-vs-storage scatter plot for one selected day."""
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    for action_class in ACTION_COLUMNS:
        action_samples = day_samples[day_samples["action_class"] == action_class]
        if action_samples.empty:
            continue
        axis.scatter(
            action_samples["price"],
            action_samples["storage_level"],
            label=action_class,
            color=ACTION_COLORS[action_class],
            alpha=0.75,
        )
    axis.set_xlabel("Price")
    axis.set_ylabel("Storage level")
    axis.set_title(f"Absolute day {absolute_day}")
    axis.legend(title="Action")
    figure.tight_layout()
    figure.savefig(
        output_dir / f"day_{absolute_day:03d}_price_vs_storage.png",
        dpi=160,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
