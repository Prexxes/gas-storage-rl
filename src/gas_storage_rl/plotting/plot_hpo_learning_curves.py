"""Plot seed-aggregated HPO learning curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gas_storage_rl.plotting.statistics import bootstrap_percentile_ci

COMPARISON_LABEL = "SB3 default"
HPO_OVERVIEW_LABEL = "HPO trials"
STEP_COLUMN = "total_training_env_steps"
VALUE_COLUMN = "mean_return_raw"


def main() -> None:
    """Creates HPO learning curve plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["overview", "best_only"],
        default="overview",
    )
    parser.add_argument("--comparison-group-dir")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir")
    parser.add_argument("--title")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else study_dir / "plots" / "hpo_learning_curves"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_seed_metrics = load_hpo_seed_evaluations(study_dir, args.split)
    comparison_seed_metrics = (
        load_experiment_group_seed_evaluations(
            Path(args.comparison_group_dir),
            args.split,
        )
        if args.comparison_group_dir
        else pd.DataFrame()
    )
    ranking = load_hpo_trial_ranking(study_dir)
    figure = plot_hpo_learning_curves(
        trial_seed_metrics,
        mode=args.mode,
        comparison_seed_metrics=comparison_seed_metrics,
        ranking=ranking,
        top_k=args.top_k,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        title=args.title,
    )
    _save(figure, output_dir / f"hpo_learning_curves_{args.mode}_{args.split}.png")
    print(f"Saved HPO learning curve plot to {output_dir.resolve()}")


def plot_hpo_learning_curves(
    trial_seed_metrics: pd.DataFrame,
    *,
    mode: str = "overview",
    comparison_seed_metrics: pd.DataFrame | None = None,
    ranking: pd.DataFrame | None = None,
    top_k: int = 3,
    n_bootstrap: int = 10_000,
    random_seed: int = 0,
    title: str | None = None,
) -> plt.Figure:
    """Plots HPO trial learning curves aggregated over seeds.

    Args:
        trial_seed_metrics: Per-seed HPO evaluation rows.
        mode: Plot mode, either ``overview`` or ``best_only``.
        comparison_seed_metrics: Optional per-seed comparison-group evaluations.
        ranking: Optional trial ranking with objective values.
        top_k: Number of best HPO trials highlighted in overview mode.
        n_bootstrap: Number of bootstrap samples for confidence bands.
        random_seed: Seed used for deterministic confidence bands.
        title: Optional plot title.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If the mode or input data are invalid.
    """
    if mode not in {"overview", "best_only"}:
        raise ValueError("mode must be 'overview' or 'best_only'")
    if trial_seed_metrics.empty:
        raise ValueError("trial_seed_metrics must not be empty")

    aggregate = aggregate_learning_curves(trial_seed_metrics)
    top_trial_ids = rank_trial_ids(aggregate, ranking)[: max(1, int(top_k))]
    figure, axis = plt.subplots(figsize=(9, 5))

    if mode == "overview":
        _plot_overview(axis, aggregate, top_trial_ids)
    else:
        _plot_best_only(
            axis,
            trial_seed_metrics,
            top_trial_ids[0],
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

    if comparison_seed_metrics is not None and not comparison_seed_metrics.empty:
        comparison_aggregate = aggregate_comparison_learning_curve(
            comparison_seed_metrics
        )
        if mode == "overview":
            axis.plot(
                comparison_aggregate[STEP_COLUMN],
                comparison_aggregate[VALUE_COLUMN],
                color="black",
                linewidth=2.8,
                label=COMPARISON_LABEL,
                zorder=4,
            )
        else:
            _plot_curve_with_ci(
                axis,
                comparison_seed_metrics,
                label=COMPARISON_LABEL,
                color="black",
                n_bootstrap=n_bootstrap,
                random_seed=random_seed + 10_000,
            )

    axis.set_xlabel("Total environment steps")
    axis.set_ylabel("Mean raw return")
    if title:
        axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    return figure


def load_hpo_seed_evaluations(study_dir: Path, split: str) -> pd.DataFrame:
    """Loads per-seed evaluation curves for all completed HPO trials.

    Args:
        study_dir: Directory containing HPO trial JSON artifacts.
        split: Dataset split name.

    Returns:
        Per-seed HPO evaluation rows.
    """
    frames = []
    for path in sorted(study_dir.glob("trial_[0-9][0-9][0-9][0-9].json")):
        payload = _read_json(path)
        if payload.get("status") != "completed":
            continue
        trial_id = int(payload["trial_id"])
        for seed_run in payload.get("seed_runs", []):
            frame = _read_run_evaluations(Path(seed_run.get("run_dir", "")), split)
            if frame.empty:
                continue
            frame["trial_id"] = trial_id
            frame["seed_index"] = int(seed_run["seed_index"])
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_experiment_group_seed_evaluations(
    group_dir: Path,
    split: str,
) -> pd.DataFrame:
    """Loads per-seed evaluation curves for one experiment group.

    Args:
        group_dir: Experiment-group directory containing ``runs.csv``.
        split: Dataset split name.

    Returns:
        Per-seed comparison evaluation rows.
    """
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


def load_hpo_trial_ranking(study_dir: Path) -> pd.DataFrame:
    """Loads HPO objective values used for ranking highlighted trials.

    Args:
        study_dir: Directory containing ``trials.csv``.

    Returns:
        Dataframe with ``trial_id`` and objective columns, if available.
    """
    path = study_dir / "trials.csv"
    if not path.exists():
        return pd.DataFrame()
    ranking = pd.read_csv(path)
    columns = ["trial_id", "objective_mean_validation_return_raw"]
    if not set(columns).issubset(ranking.columns):
        return pd.DataFrame()
    if "status" in ranking.columns:
        ranking = ranking[ranking["status"] == "completed"].copy()
    ranking["trial_id"] = ranking["trial_id"].astype(int)
    ranking["objective_mean_validation_return_raw"] = pd.to_numeric(
        ranking["objective_mean_validation_return_raw"],
        errors="coerce",
    )
    return ranking.dropna(subset=["objective_mean_validation_return_raw"])[columns]


def aggregate_learning_curves(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregates per-seed HPO evaluation rows by trial and evaluation step.

    Args:
        seed_metrics: Per-seed evaluation rows with a ``trial_id`` column.

    Returns:
        Trial-level mean learning curves.
    """
    required = {"trial_id", STEP_COLUMN, VALUE_COLUMN}
    _require_columns(seed_metrics, required)
    data = seed_metrics.copy()
    data[STEP_COLUMN] = pd.to_numeric(data[STEP_COLUMN])
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN])
    aggregate = (
        data.groupby(["trial_id", STEP_COLUMN], as_index=False)[VALUE_COLUMN]
        .mean()
        .sort_values(["trial_id", STEP_COLUMN])
    )
    aggregate["trial_id"] = aggregate["trial_id"].astype(int)
    return aggregate


def aggregate_comparison_learning_curve(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregates comparison-group evaluations by evaluation step.

    Args:
        seed_metrics: Per-seed evaluation rows.

    Returns:
        Seed-aggregated comparison learning curve.
    """
    required = {STEP_COLUMN, VALUE_COLUMN}
    _require_columns(seed_metrics, required)
    data = seed_metrics.copy()
    data[STEP_COLUMN] = pd.to_numeric(data[STEP_COLUMN])
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN])
    return (
        data.groupby(STEP_COLUMN, as_index=False)[VALUE_COLUMN]
        .mean()
        .sort_values(STEP_COLUMN)
    )


def rank_trial_ids(
    aggregate: pd.DataFrame,
    ranking: pd.DataFrame | None = None,
) -> list[int]:
    """Returns trial ids ordered from best to worst.

    Args:
        aggregate: Aggregated HPO learning curves.
        ranking: Optional HPO objective ranking.

    Returns:
        Ordered trial ids.
    """
    if ranking is not None and not ranking.empty:
        ordered = ranking.sort_values(
            "objective_mean_validation_return_raw",
            ascending=False,
        )
        return [int(trial_id) for trial_id in ordered["trial_id"]]

    final_rows = aggregate.sort_values(STEP_COLUMN).groupby("trial_id").tail(1)
    final_rows = final_rows.sort_values(VALUE_COLUMN, ascending=False)
    return [int(trial_id) for trial_id in final_rows["trial_id"]]


def _plot_overview(
    axis: plt.Axes,
    aggregate: pd.DataFrame,
    top_trial_ids: list[int],
) -> None:
    """Plots all HPO curves lightly and highlights top trials."""
    top_trial_ids = top_trial_ids[:3]
    for trial_id, group in aggregate.groupby("trial_id", sort=False):
        group = group.sort_values(STEP_COLUMN)
        if int(trial_id) in top_trial_ids:
            continue
        axis.plot(
            group[STEP_COLUMN],
            group[VALUE_COLUMN],
            color="0.45",
            alpha=0.18,
            linewidth=0.9,
            zorder=1,
        )
    highlighted_colors = ["tab:blue", "tab:orange", "tab:green"]
    for rank, trial_id in enumerate(top_trial_ids, start=1):
        group = aggregate[aggregate["trial_id"] == trial_id].sort_values(STEP_COLUMN)
        axis.plot(
            group[STEP_COLUMN],
            group[VALUE_COLUMN],
            color=highlighted_colors[rank - 1],
            linewidth=2.4,
            label=f"Top {rank}: Trial {trial_id}",
            zorder=3,
        )
    if len(aggregate["trial_id"].unique()) > len(top_trial_ids):
        axis.plot([], [], color="0.45", alpha=0.35, linewidth=1.2, label=HPO_OVERVIEW_LABEL)


def _plot_best_only(
    axis: plt.Axes,
    seed_metrics: pd.DataFrame,
    trial_id: int,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> None:
    """Plots the best HPO trial with a bootstrap confidence band."""
    data = seed_metrics[seed_metrics["trial_id"] == trial_id]
    _plot_curve_with_ci(
        axis,
        data,
        label=f"Best HPO trial {trial_id}",
        color="tab:blue",
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )


def _plot_curve_with_ci(
    axis: plt.Axes,
    seed_metrics: pd.DataFrame,
    *,
    label: str,
    color: str,
    n_bootstrap: int,
    random_seed: int,
) -> None:
    """Plots a seed-aggregated curve with 95% bootstrap percentile intervals."""
    aggregate = aggregate_comparison_learning_curve(seed_metrics)
    intervals = _bootstrap_curve_intervals(
        seed_metrics,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
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
        aggregate[VALUE_COLUMN],
        color=color,
        linewidth=2.6,
        label=label,
        zorder=4,
    )


def _bootstrap_curve_intervals(
    seed_metrics: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    """Computes bootstrap intervals for each evaluation step."""
    rows = []
    data = seed_metrics.copy()
    data[STEP_COLUMN] = pd.to_numeric(data[STEP_COLUMN])
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN])
    for offset, (step, group) in enumerate(data.groupby(STEP_COLUMN, sort=True)):
        lower, upper = bootstrap_percentile_ci(
            group[VALUE_COLUMN].to_numpy(),
            confidence_level=0.95,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + offset,
        )
        rows.append({STEP_COLUMN: step, "ci_lower": lower, "ci_upper": upper})
    return pd.DataFrame(rows)


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


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    """Raises a clear error when required columns are missing."""
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _read_json(path: Path) -> dict[str, Any]:
    """Reads a JSON document."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save(figure: plt.Figure, path: Path) -> None:
    """Saves and closes a Matplotlib figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
