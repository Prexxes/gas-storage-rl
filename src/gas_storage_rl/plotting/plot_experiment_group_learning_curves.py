"""Plot seed-aggregated learning curves for experiment groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gas_storage_rl.evaluation.metrics import interquartile_mean
from gas_storage_rl.plotting.statistics import bootstrap_percentile_statistic_ci

STEP_COLUMN = "total_training_env_steps"
VALUE_COLUMN = "mean_return_raw"
AGGREGATE_VALUE_COLUMN = "mean_return_raw_over_seed"
DEFAULT_Y_LABEL = "Mittlerer operativer Return über Seeds"
INTERQUARTILE_MEAN_Y_LABEL = "Interquartile Mean Return über Seeds"
BENCHMARK_LINESTYLE = "--"
SEED_AGGREGATES = {"mean", "interquartile_mean"}
BENCHMARK_DISPLAY_LABELS = {
    "random": "Random Policy",
    "rule_based": "Rule-based Policy",
    "perfect_foresight": "Perfect Foresight",
    "oracle_cloned_policy": "Oracle-Cloned Policy",
    "lsmc": "LSMC",
}


def main() -> None:
    """Creates seed-aggregated experiment-group learning curve plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-group-dir", action="append", required=True)
    parser.add_argument("--group-label", action="append", default=[])
    parser.add_argument("--benchmark-run-dir")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir")
    parser.add_argument("--title")
    parser.add_argument(
        "--environment-label",
        help="Environment label for the default title.",
    )
    parser.add_argument(
        "--capacity",
        help="Storage capacity label for the default title.",
    )
    parser.add_argument(
        "--seed-aggregate",
        choices=sorted(SEED_AGGREGATES),
        default="mean",
    )
    parser.add_argument("--y-label")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    group_dirs = [Path(group_dir) for group_dir in args.experiment_group_dir]
    if args.group_label and len(args.group_label) != len(group_dirs):
        raise ValueError("--group-label must be passed once per experiment group")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else group_dirs[0] / "plots" / "experiment_group_learning_curves"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    group_metrics = load_experiment_group_learning_curves(
        group_dirs,
        split=args.split,
        group_labels=args.group_label,
    )
    benchmark_metrics = (
        load_benchmark_learning_curves(Path(args.benchmark_run_dir), args.split)
        if args.benchmark_run_dir
        else pd.DataFrame()
    )
    figure = plot_experiment_group_learning_curves(
        group_metrics,
        benchmark_metrics=benchmark_metrics,
        title=build_default_title(
            environment_label=args.environment_label,
            capacity=args.capacity,
        )
        if args.title is None
        else args.title,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        seed_aggregate=args.seed_aggregate,
        y_label=args.y_label,
    )
    output_path = (
        output_dir
        / f"learning_curves_experiment_groups_{args.split}_{args.seed_aggregate}.png"
    )
    _save(figure, output_path)
    print(f"Saved experiment-group learning curve plot to {output_dir.resolve()}")


def plot_experiment_group_learning_curves(
    group_metrics: pd.DataFrame,
    *,
    benchmark_metrics: pd.DataFrame | None = None,
    title: str | None = None,
    n_bootstrap: int = 10_000,
    random_seed: int = 0,
    seed_aggregate: str = "mean",
    y_label: str | None = None,
) -> plt.Figure:
    """Plots experiment-group learning curves with bootstrap confidence bands.

    Args:
        group_metrics: Per-seed evaluation rows with group labels.
        benchmark_metrics: Optional benchmark reference rows.
        title: Optional plot title.
        n_bootstrap: Number of bootstrap samples for confidence bands.
        random_seed: Seed used for deterministic confidence bands.
        seed_aggregate: Statistic used to aggregate seed values by step.
        y_label: Optional y-axis label.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If required input data are missing.
    """
    if group_metrics.empty:
        raise ValueError("group_metrics must not be empty")
    _require_columns(group_metrics, {"group_label", STEP_COLUMN, VALUE_COLUMN})
    _validate_seed_aggregate(seed_aggregate)

    figure, axis = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    grouped_metrics = group_metrics.groupby("group_label", sort=False)
    for offset, (label, group) in enumerate(grouped_metrics):
        color = colors[offset % len(colors)]
        _plot_group_curve_with_ci(
            axis,
            group,
            label=str(label),
            color=color,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + offset * 10_000,
            seed_aggregate=seed_aggregate,
        )

    if benchmark_metrics is not None and not benchmark_metrics.empty:
        _plot_benchmark_references(axis, benchmark_metrics)

    axis.set_xlabel("Trainingsschritte")
    axis.set_ylabel(y_label or _default_y_label(seed_aggregate))
    if title:
        axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    return figure


def load_experiment_group_learning_curves(
    group_dirs: list[Path],
    *,
    split: str,
    group_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Loads per-seed evaluation curves for experiment groups.

    Args:
        group_dirs: Experiment-group directories containing ``runs.csv``.
        split: Dataset split name.
        group_labels: Optional display labels, one per group directory.

    Returns:
        Per-seed evaluation rows for all groups.
    """
    labels = group_labels or []
    frames = []
    for index, group_dir in enumerate(group_dirs):
        label = labels[index] if labels else _default_group_label(group_dir)
        frame = _load_one_experiment_group(group_dir, split)
        if frame.empty:
            continue
        frame["group_label"] = label
        frame["experiment_group_dir"] = str(group_dir)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_benchmark_learning_curves(
    benchmark_run_dir: Path,
    split: str,
) -> pd.DataFrame:
    """Loads benchmark reference learning-curve rows.

    Args:
        benchmark_run_dir: Benchmark run directory.
        split: Dataset split name.

    Returns:
        Benchmark reference rows for the requested split.
    """
    path = benchmark_run_dir / "benchmark_evaluations.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "split" in frame.columns:
        frame = frame[frame["split"] == split].copy()
    return frame


def aggregate_experiment_group_learning_curve(
    seed_metrics: pd.DataFrame,
    *,
    seed_aggregate: str = "mean",
) -> pd.DataFrame:
    """Aggregates per-seed returns by evaluation step.

    Args:
        seed_metrics: Per-seed evaluation rows.
        seed_aggregate: Statistic used to aggregate seed values by step.

    Returns:
        Stepwise seed-averaged learning curve.
    """
    _require_columns(seed_metrics, {STEP_COLUMN, VALUE_COLUMN})
    _validate_seed_aggregate(seed_aggregate)
    data = deduplicate_seed_step_evaluations(seed_metrics)
    data[STEP_COLUMN] = pd.to_numeric(data[STEP_COLUMN])
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN])
    aggregate_column = _aggregate_value_column(seed_aggregate)
    rows = []
    for step, group in data.groupby(STEP_COLUMN, sort=True):
        rows.append(
            {
                STEP_COLUMN: step,
                aggregate_column: _seed_statistic(
                    group[VALUE_COLUMN].to_numpy(),
                    seed_aggregate,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_default_title(
    *,
    environment_label: str | None = None,
    capacity: str | None = None,
) -> str:
    """Builds the German default title from optional plot context.

    Args:
        environment_label: Environment display label.
        capacity: Storage capacity display value.

    Returns:
        Default plot title.
    """
    if environment_label and capacity:
        return (
            f"Lernkurven auf dem {environment_label}-Environment "
            f"mit Speicherkapazität {capacity}"
        )
    if environment_label:
        return f"Lernkurven auf dem {environment_label}-Environment"
    if capacity:
        return f"Lernkurven mit Speicherkapazität {capacity}"
    return "Lernkurven der RL-Algorithmen"


def _benchmark_display_label(method: str) -> str:
    """Returns the display label for known benchmark methods."""
    return BENCHMARK_DISPLAY_LABELS.get(method, method)


def deduplicate_seed_step_evaluations(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Keeps the last evaluation row for each seed and evaluation step.

    Args:
        seed_metrics: Per-seed evaluation rows.

    Returns:
        Evaluation rows with at most one row per seed and step.
    """
    _require_columns(seed_metrics, {"seed_index", STEP_COLUMN})
    data = seed_metrics.copy()
    data[STEP_COLUMN] = pd.to_numeric(data[STEP_COLUMN])
    return data.drop_duplicates(["seed_index", STEP_COLUMN], keep="last")


def _plot_group_curve_with_ci(
    axis: plt.Axes,
    seed_metrics: pd.DataFrame,
    *,
    label: str,
    color: str,
    n_bootstrap: int,
    random_seed: int,
    seed_aggregate: str,
) -> None:
    """Plots one seed-aggregated group curve with a confidence band."""
    aggregate = aggregate_experiment_group_learning_curve(
        seed_metrics,
        seed_aggregate=seed_aggregate,
    )
    intervals = _bootstrap_curve_intervals(
        seed_metrics,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        seed_aggregate=seed_aggregate,
    )
    aggregate_column = _aggregate_value_column(seed_aggregate)
    axis.fill_between(
        intervals[STEP_COLUMN].to_numpy(),
        intervals["ci_lower"].to_numpy(),
        intervals["ci_upper"].to_numpy(),
        color=color,
        alpha=0.18,
        linewidth=0.0,
    )
    axis.plot(
        aggregate[STEP_COLUMN],
        aggregate[aggregate_column],
        color=color,
        linewidth=2.5,
        label=label,
        zorder=4,
    )


def _plot_benchmark_references(
    axis: plt.Axes,
    benchmark_metrics: pd.DataFrame,
) -> None:
    """Plots benchmark reference lines."""
    _require_columns(benchmark_metrics, {"method", STEP_COLUMN, VALUE_COLUMN})
    for method, group in benchmark_metrics.groupby("method", sort=False):
        group = group.sort_values(STEP_COLUMN)
        axis.plot(
            group[STEP_COLUMN],
            group[VALUE_COLUMN],
            label=_benchmark_display_label(str(method)),
            linestyle=BENCHMARK_LINESTYLE,
            linewidth=1.8,
            alpha=0.85,
        )


def _bootstrap_curve_intervals(
    seed_metrics: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
    seed_aggregate: str,
) -> pd.DataFrame:
    """Computes bootstrap intervals for each evaluation step."""
    _validate_seed_aggregate(seed_aggregate)
    rows = []
    data = deduplicate_seed_step_evaluations(seed_metrics)
    data[STEP_COLUMN] = pd.to_numeric(data[STEP_COLUMN])
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN])
    for offset, (step, group) in enumerate(data.groupby(STEP_COLUMN, sort=True)):
        lower, upper = bootstrap_percentile_statistic_ci(
            group[VALUE_COLUMN].to_numpy(),
            statistic=lambda values: _seed_statistic(values, seed_aggregate),
            confidence_level=0.95,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + offset,
        )
        rows.append({STEP_COLUMN: step, "ci_lower": lower, "ci_upper": upper})
    return pd.DataFrame(rows)


def _seed_statistic(values: np.ndarray, seed_aggregate: str) -> float:
    """Computes the selected seed aggregation statistic."""
    if seed_aggregate == "mean":
        return float(np.mean(values))
    if seed_aggregate == "interquartile_mean":
        return interquartile_mean(values)
    raise ValueError(f"Unknown seed_aggregate: {seed_aggregate}")


def _aggregate_value_column(seed_aggregate: str) -> str:
    """Returns the aggregate value column name for the selected statistic."""
    if seed_aggregate == "mean":
        return AGGREGATE_VALUE_COLUMN
    if seed_aggregate == "interquartile_mean":
        return "interquartile_mean_return_raw_over_seed"
    raise ValueError(f"Unknown seed_aggregate: {seed_aggregate}")


def _default_y_label(seed_aggregate: str) -> str:
    """Returns the default y-axis label for the selected seed aggregate."""
    if seed_aggregate == "mean":
        return DEFAULT_Y_LABEL
    if seed_aggregate == "interquartile_mean":
        return INTERQUARTILE_MEAN_Y_LABEL
    raise ValueError(f"Unknown seed_aggregate: {seed_aggregate}")


def _validate_seed_aggregate(seed_aggregate: str) -> None:
    """Raises a clear error for unknown seed aggregation modes."""
    if seed_aggregate not in SEED_AGGREGATES:
        raise ValueError(f"seed_aggregate must be one of {sorted(SEED_AGGREGATES)}")


def _load_one_experiment_group(group_dir: Path, split: str) -> pd.DataFrame:
    """Loads per-seed evaluations from one experiment-group manifest."""
    manifest_path = group_dir / "runs.csv"
    if not manifest_path.exists():
        return pd.DataFrame()
    runs = pd.read_csv(manifest_path)
    frames = []
    for _, row in runs.iterrows():
        if str(row.get("status", "")) not in {"completed", "skipped"}:
            continue
        frame = _read_run_evaluations(Path(str(row.get("run_dir", ""))), split)
        if frame.empty:
            continue
        frame["seed_index"] = int(row["seed_index"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_run_evaluations(run_dir: Path, split: str) -> pd.DataFrame:
    """Reads one run's evaluation CSV for the requested split."""
    path = run_dir / "evaluations.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "split" in frame.columns:
        frame = frame[frame["split"] == split].copy()
    if STEP_COLUMN not in frame.columns and "total_env_steps" in frame.columns:
        frame = frame.rename(columns={"total_env_steps": STEP_COLUMN})
    return frame[[STEP_COLUMN, VALUE_COLUMN]].copy()


def _default_group_label(group_dir: Path) -> str:
    """Returns the default display label for an experiment group."""
    manifest_path = group_dir / "runs.csv"
    if manifest_path.exists():
        runs = pd.read_csv(manifest_path)
        if "algorithm_name" in runs.columns:
            algorithms = runs["algorithm_name"].dropna().astype(str).unique()
            if len(algorithms) == 1:
                return algorithms[0].upper()
    return group_dir.name


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    """Raises a clear error when required columns are missing."""
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _save(figure: plt.Figure, path: Path) -> None:
    """Saves and closes a Matplotlib figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
