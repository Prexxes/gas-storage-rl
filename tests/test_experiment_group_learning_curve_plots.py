"""Tests for experiment-group learning curve plotting helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gas_storage_rl.plotting.plot_experiment_group_learning_curves import (
    AGGREGATE_VALUE_COLUMN,
    BENCHMARK_LINESTYLE,
    DEFAULT_Y_LABEL,
    aggregate_experiment_group_learning_curve,
    build_default_title,
    deduplicate_seed_step_evaluations,
    load_experiment_group_learning_curves,
    plot_experiment_group_learning_curves,
)


def test_aggregate_experiment_group_learning_curve_uses_seed_mean() -> None:
    """The plotted curve averages mean_return_raw over seeds by step."""
    metrics = pd.DataFrame(
        {
            "seed_index": [0, 0, 1, 1],
            "total_training_env_steps": [0, 10, 0, 10],
            "mean_return_raw": [1.0, 3.0, 5.0, 7.0],
        }
    )

    aggregate = aggregate_experiment_group_learning_curve(metrics)

    assert aggregate[AGGREGATE_VALUE_COLUMN].tolist() == [3.0, 5.0]


def test_build_default_title_uses_environment_and_capacity() -> None:
    """The German default title uses Umgebung and storage capacity."""
    title = build_default_title(environment_label="OU", capacity="30")

    assert title == "Lernkurven auf der OU-Umgebung mit Kapazität 30"


def test_deduplicate_seed_step_evaluations_keeps_last_row() -> None:
    """Duplicate final callback rows do not overweight a seed in the plot."""
    metrics = pd.DataFrame(
        {
            "seed_index": [0, 0, 1],
            "total_training_env_steps": [10, 10, 10],
            "mean_return_raw": [1.0, 3.0, 5.0],
        }
    )

    deduplicated = deduplicate_seed_step_evaluations(metrics)
    aggregate = aggregate_experiment_group_learning_curve(metrics)

    assert deduplicated["mean_return_raw"].tolist() == [3.0, 5.0]
    assert aggregate[AGGREGATE_VALUE_COLUMN].tolist() == [4.0]


def test_plot_uses_short_y_label_title_ci_and_benchmark_line() -> None:
    """Learning curve plot contains seed CI bands and optional benchmarks."""
    group_metrics = pd.DataFrame(
        {
            "group_label": ["PPO", "PPO", "PPO", "PPO"],
            "seed_index": [0, 0, 1, 1],
            "total_training_env_steps": [0, 10, 0, 10],
            "mean_return_raw": [1.0, 3.0, 5.0, 7.0],
        }
    )
    benchmark_metrics = pd.DataFrame(
        {
            "method": ["lsmc", "lsmc"],
            "total_training_env_steps": [0, 10],
            "mean_return_raw": [4.0, 4.0],
        }
    )

    figure = plot_experiment_group_learning_curves(
        group_metrics,
        benchmark_metrics=benchmark_metrics,
        title="Lernkurven auf der OU-Umgebung mit Kapazität 30",
        n_bootstrap=200,
    )
    axis = figure.axes[0]
    labels = [text.get_text() for text in axis.get_legend().get_texts()]

    assert axis.get_ylabel() == DEFAULT_Y_LABEL
    assert axis.get_title() == "Lernkurven auf der OU-Umgebung mit Kapazität 30"
    assert labels == ["PPO", "lsmc"]
    assert len(axis.collections) == 1
    assert axis.lines[1].get_linestyle() == BENCHMARK_LINESTYLE
    plt.close(figure)


def test_load_experiment_group_learning_curves_reads_completed_seed_runs(
    tmp_path: Path,
) -> None:
    """Experiment-group loading reads runs.csv and filters by split."""
    group_dir = tmp_path / "group"
    run_dir = tmp_path / "run-0"
    group_dir.mkdir()
    run_dir.mkdir()
    _write_csv(
        group_dir / "runs.csv",
        [
            {
                "status": "completed",
                "seed_index": "0",
                "run_dir": str(run_dir),
            }
        ],
    )
    _write_csv(
        run_dir / "evaluations.csv",
        [
            {
                "split": "train",
                "total_training_env_steps": "0",
                "mean_return_raw": "1.0",
            },
            {
                "split": "validation",
                "total_training_env_steps": "0",
                "mean_return_raw": "2.0",
            },
        ],
    )

    metrics = load_experiment_group_learning_curves(
        [group_dir],
        split="validation",
        group_labels=["PPO"],
    )

    assert metrics["group_label"].tolist() == ["PPO"]
    assert metrics["mean_return_raw"].tolist() == [2.0]


def test_load_experiment_group_learning_curves_defaults_label_from_algorithm_name(
    tmp_path: Path,
) -> None:
    """Default labels use algorithm_name from the experiment-group manifest."""
    group_dir = tmp_path / "long-group-id"
    run_dir = tmp_path / "run-0"
    group_dir.mkdir()
    run_dir.mkdir()
    _write_csv(
        group_dir / "runs.csv",
        [
            {
                "algorithm_name": "td3",
                "status": "completed",
                "seed_index": "0",
                "run_dir": str(run_dir),
            }
        ],
    )
    _write_csv(
        run_dir / "evaluations.csv",
        [
            {
                "split": "validation",
                "total_training_env_steps": "0",
                "mean_return_raw": "2.0",
            },
        ],
    )

    metrics = load_experiment_group_learning_curves(
        [group_dir],
        split="validation",
    )

    assert metrics["group_label"].tolist() == ["TD3"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Writes a small CSV fixture."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
