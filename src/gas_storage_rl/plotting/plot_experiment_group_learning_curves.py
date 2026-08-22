"""Plot seed-aggregated learning curves for experiment groups."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
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
BENCHMARK_COLORS = {
    "random": "#7f7f7f",                # gray
    "rule_based": "#8c564b",            # brown
    "perfect_foresight": "#d62728",     # red
    "oracle_cloned_policy": "#000000",  # black
    "lsmc": "#9467bd",                  # purple
}
SEED_AGGREGATES = {"mean", "interquartile_mean"}
Y_AXIS_SCALE_MODES = {"all", "none", "row"}
BENCHMARK_DISPLAY_LABELS = {
    "random": "Random Policy",
    "rule_based": "Rule-based Policy",
    "perfect_foresight": "Perfect Foresight",
    "oracle_cloned_policy": "Oracle-Cloned Policy",
    "lsmc": "LSMC",
}
GRID_DEFAULT_TITLE = "Lernkurven nach Environment und Speicherkapazität"
GRID_PANEL_SPECS = (
    ("deterministic_c200", "deterministic-c200", "Deterministic", "200"),
    ("ou_c200", "ou-c200", "OU", "200"),
    ("deterministic_c30", "deterministic-c30", "Deterministic", "30"),
    ("ou_c30", "ou-c30", "OU", "30"),
)


@dataclass(frozen=True)
class LearningCurveGridPanel:
    """Input data and labels for one learning-curve grid panel."""

    environment_label: str
    capacity: str
    group_metrics: pd.DataFrame
    benchmark_metrics: pd.DataFrame | None = None
    title: str | None = None


def main() -> None:
    """Creates seed-aggregated experiment-group learning curve plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-group-dir", action="append", default=[])
    parser.add_argument("--group-label", action="append", default=[])
    parser.add_argument("--benchmark-run-dir")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir")
    parser.add_argument("--title")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Create the fixed 2x2 Deterministic/OU and C=200/C=30 grid.",
    )
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
    parser.add_argument(
        "--share-y",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Deprecated grid y-axis shortcut. --share-y uses one shared scale; "
            "--no-share-y uses independent panel scales."
        ),
    )
    parser.add_argument(
        "--y-axis-scale",
        choices=sorted(Y_AXIS_SCALE_MODES),
        default="row",
        help=(
            "Grid y-axis scaling: row shares one scale per row, all shares one "
            "scale for all panels, none uses independent panel scales."
        ),
    )
    _add_grid_panel_arguments(parser)
    args = parser.parse_args()

    if args.grid or _has_grid_panel_args(args):
        _run_grid_plot(args)
        return
    _run_single_plot(args)


def _run_single_plot(args: argparse.Namespace) -> None:
    """Creates the existing single-panel learning-curve plot."""
    if not args.experiment_group_dir:
        raise ValueError("--experiment-group-dir is required for single-panel plots")
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


def _run_grid_plot(args: argparse.Namespace) -> None:
    """Creates the fixed 2x2 learning-curve grid plot."""
    panels = _load_grid_panels(args)
    first_group_dirs = _grid_panel_group_dirs(args, GRID_PANEL_SPECS[0][0])
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else first_group_dirs[0] / "plots" / "experiment_group_learning_curves"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plot_experiment_group_learning_curve_grid(
        panels,
        title=args.title or GRID_DEFAULT_TITLE,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        seed_aggregate=args.seed_aggregate,
        y_label=args.y_label,
        y_axis_scale=_resolve_y_axis_scale(args),
    )
    output_path = (
        output_dir
        / (
            "learning_curves_experiment_group_grid_"
            f"{args.split}_{args.seed_aggregate}.png"
        )
    )
    _save(figure, output_path, tight_layout=False)
    print(f"Saved experiment-group learning curve grid to {output_dir.resolve()}")


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
    _validate_seed_aggregate(seed_aggregate)

    figure, axis = plt.subplots(figsize=(9, 5))
    _plot_learning_curves_on_axis(
        axis,
        group_metrics,
        benchmark_metrics=benchmark_metrics,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        seed_aggregate=seed_aggregate,
    )
    axis.set_xlabel("Trainingsschritte")
    axis.set_ylabel(y_label or _default_y_label(seed_aggregate))
    if title:
        axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, handlelength = 5.0)
    return figure


def plot_experiment_group_learning_curve_grid(
    panels: list[LearningCurveGridPanel],
    *,
    title: str | None = GRID_DEFAULT_TITLE,
    n_bootstrap: int = 10_000,
    random_seed: int = 0,
    seed_aggregate: str = "mean",
    y_label: str | None = None,
    y_axis_scale: str | bool = "row",
    share_y: bool | None = None,
) -> plt.Figure:
    """Plots a fixed 2x2 learning-curve grid with one shared legend.

    The panel order is upper-left, upper-right, lower-left, lower-right.

    Args:
        panels: Four panel inputs in display order.
        title: Optional figure title.
        n_bootstrap: Number of bootstrap samples for confidence bands.
        random_seed: Seed used for deterministic confidence bands.
        seed_aggregate: Statistic used to aggregate seed values by step.
        y_label: Optional shared y-axis label.
        y_axis_scale: Y-axis scale sharing mode: ``row``, ``all``, or ``none``.
            Boolean values are accepted for backward compatibility.
        share_y: Deprecated boolean y-axis sharing override.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If the grid does not receive exactly four panels.
    """
    if len(panels) != 4:
        raise ValueError("Exactly four panels are required for the 2x2 grid")
    _validate_seed_aggregate(seed_aggregate)
    if share_y is not None:
        y_axis_scale = share_y
    share_y = _matplotlib_share_y_mode(y_axis_scale)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        sharex=True,
        sharey=share_y,
    )
    for panel_index, (axis, panel) in enumerate(zip(axes.flat, panels)):
        _plot_learning_curves_on_axis(
            axis,
            panel.group_metrics,
            benchmark_metrics=panel.benchmark_metrics,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + panel_index * 100_000,
            seed_aggregate=seed_aggregate,
        )
        axis.set_title(panel.title or build_panel_title(panel))
        axis.grid(alpha=0.25)

    handles, labels = _unique_legend_entries(axes.flat)
    figure.supxlabel("Trainingsschritte", y=0.07)
    figure.supylabel(y_label or _default_y_label(seed_aggregate), x=0.01)
    if title:
        figure.suptitle(title)
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(labels), 5),
            fontsize=8,
            handlelength = 5.0,
        )
    top = 0.94 if title else 0.98
    figure.tight_layout(rect=(0.03, 0.12, 1.0, top))
    return figure


def build_panel_title(panel: LearningCurveGridPanel) -> str:
    """Builds a concise title for one learning-curve grid panel."""
    return f"{panel.environment_label}, Speicherkapazität {panel.capacity}"


def _add_grid_panel_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds fixed 2x2 grid panel input arguments to the CLI parser."""
    for dest_prefix, cli_prefix, environment_label, capacity in GRID_PANEL_SPECS:
        panel_label = f"{environment_label} with storage capacity {capacity}"
        parser.add_argument(
            f"--{cli_prefix}-experiment-group-dir",
            action="append",
            default=[],
            dest=f"{dest_prefix}_experiment_group_dir",
            help=f"Experiment group directory for the {panel_label} panel.",
        )
        parser.add_argument(
            f"--{cli_prefix}-benchmark-run-dir",
            dest=f"{dest_prefix}_benchmark_run_dir",
            help=f"Benchmark run directory for the {panel_label} panel.",
        )


def _has_grid_panel_args(args: argparse.Namespace) -> bool:
    """Returns whether any fixed-grid panel input was passed."""
    for dest_prefix, _, _, _ in GRID_PANEL_SPECS:
        if _grid_panel_group_dirs(args, dest_prefix):
            return True
        if getattr(args, f"{dest_prefix}_benchmark_run_dir"):
            return True
    return False


def _resolve_y_axis_scale(args: argparse.Namespace) -> str:
    """Resolves the grid y-axis sharing mode from current and legacy CLI flags."""
    if args.share_y is True:
        return "all"
    if args.share_y is False:
        return "none"
    return args.y_axis_scale


def _load_grid_panels(args: argparse.Namespace) -> list[LearningCurveGridPanel]:
    """Loads all fixed-grid panels from CLI arguments."""
    panels = []
    for dest_prefix, _, environment_label, capacity in GRID_PANEL_SPECS:
        group_dirs = _grid_panel_group_dirs(args, dest_prefix)
        if not group_dirs:
            raise ValueError(
                f"--{dest_prefix.replace('_', '-')}-experiment-group-dir is "
                "required for grid plots"
            )
        if args.group_label and len(args.group_label) != len(group_dirs):
            raise ValueError(
                "--group-label must be passed once per experiment group in "
                "each grid panel"
            )
        benchmark_run_dir = getattr(args, f"{dest_prefix}_benchmark_run_dir")
        panels.append(
            LearningCurveGridPanel(
                environment_label=environment_label,
                capacity=capacity,
                group_metrics=load_experiment_group_learning_curves(
                    group_dirs,
                    split=args.split,
                    group_labels=args.group_label,
                ),
                benchmark_metrics=(
                    load_benchmark_learning_curves(
                        Path(benchmark_run_dir),
                        args.split,
                    )
                    if benchmark_run_dir
                    else pd.DataFrame()
                ),
            )
        )
    return panels


def _grid_panel_group_dirs(
    args: argparse.Namespace,
    dest_prefix: str,
) -> list[Path]:
    """Returns configured experiment group directories for one grid panel."""
    return [
        Path(group_dir)
        for group_dir in getattr(args, f"{dest_prefix}_experiment_group_dir")
    ]


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


def _plot_learning_curves_on_axis(
    axis: plt.Axes,
    group_metrics: pd.DataFrame,
    *,
    benchmark_metrics: pd.DataFrame | None,
    n_bootstrap: int,
    random_seed: int,
    seed_aggregate: str,
) -> None:
    """Plots experiment-group curves and benchmark references on one axis."""
    if group_metrics.empty:
        raise ValueError("group_metrics must not be empty")
    _require_columns(group_metrics, {"group_label", STEP_COLUMN, VALUE_COLUMN})
    _validate_seed_aggregate(seed_aggregate)

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
        method_key = str(method)
        axis.plot(
            group[STEP_COLUMN],
            group[VALUE_COLUMN],
            label=_benchmark_display_label(str(method)),
            color=BENCHMARK_COLORS.get(method_key, "0.35"),
            linestyle=BENCHMARK_LINESTYLE,
            linewidth=1.8,
            alpha=0.85,
        )


def _unique_legend_entries(axes: Iterable[plt.Axes]) -> tuple[list[object], list[str]]:
    """Returns first-seen legend handles and labels across axes."""
    handles_by_label = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in handles_by_label:
                handles_by_label[label] = handle
    return list(handles_by_label.values()), list(handles_by_label.keys())


def _matplotlib_share_y_mode(y_axis_scale: str | bool) -> bool | str:
    """Converts a public y-axis scale mode into a Matplotlib sharey value."""
    if y_axis_scale is True:
        return True
    if y_axis_scale is False:
        return False
    if y_axis_scale == "all":
        return True
    if y_axis_scale == "none":
        return False
    if y_axis_scale == "row":
        return "row"
    raise ValueError(f"y_axis_scale must be one of {sorted(Y_AXIS_SCALE_MODES)}")


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


def _save(figure: plt.Figure, path: Path, *, tight_layout: bool = True) -> None:
    """Saves and closes a Matplotlib figure."""
    if tight_layout:
        figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
