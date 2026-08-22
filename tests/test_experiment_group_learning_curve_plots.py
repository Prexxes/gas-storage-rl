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
    INTERQUARTILE_MEAN_Y_LABEL,
    LearningCurveGridPanel,
    aggregate_experiment_group_learning_curve,
    build_default_title,
    deduplicate_seed_step_evaluations,
    load_experiment_group_learning_curves,
    plot_experiment_group_learning_curve_grid,
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


def test_aggregate_experiment_group_learning_curve_uses_seed_iqm() -> None:
    """IQM aggregation computes the interquartile mean over seed values."""
    metrics = pd.DataFrame(
        {
            "seed_index": [0, 1, 2, 3],
            "total_training_env_steps": [10, 10, 10, 10],
            "mean_return_raw": [0.0, 1.0, 2.0, 100.0],
        }
    )

    aggregate = aggregate_experiment_group_learning_curve(
        metrics,
        seed_aggregate="interquartile_mean",
    )

    assert aggregate["interquartile_mean_return_raw_over_seed"].tolist() == [1.5]


def test_build_default_title_uses_environment_and_capacity() -> None:
    """The German default title uses the environment and storage capacity labels."""
    title = build_default_title(environment_label="OU", capacity="30")

    assert title == "Lernkurven auf dem OU-Environment mit Speicherkapazität 30"


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


def test_plot_uses_labels_title_ci_and_benchmark_display_names() -> None:
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
            "method": [
                "random",
                "rule_based",
                "perfect_foresight",
                "oracle_cloned_policy",
                "lsmc",
            ],
            "total_training_env_steps": [10, 10, 10, 10, 10],
            "mean_return_raw": [1.0, 2.0, 8.0, 6.0, 4.0],
        }
    )

    figure = plot_experiment_group_learning_curves(
        group_metrics,
        benchmark_metrics=benchmark_metrics,
        title="Lernkurven auf dem OU-Environment mit Speicherkapazität 30",
        n_bootstrap=200,
    )
    axis = figure.axes[0]
    labels = [text.get_text() for text in axis.get_legend().get_texts()]

    assert axis.get_xlabel() == "Trainingsschritte"
    assert axis.get_ylabel() == DEFAULT_Y_LABEL
    assert axis.get_title() == (
        "Lernkurven auf dem OU-Environment mit Speicherkapazität 30"
    )
    assert labels == [
        "PPO",
        "Random Policy",
        "Rule-based Policy",
        "Perfect Foresight",
        "Oracle-Cloned Policy",
        "LSMC",
    ]
    assert len(axis.collections) == 1
    assert axis.lines[1].get_linestyle() == BENCHMARK_LINESTYLE
    plt.close(figure)


def test_plot_iqm_uses_matching_y_label_and_curve_statistic() -> None:
    """The IQM plot line and y-axis use the selected seed aggregate."""
    group_metrics = pd.DataFrame(
        {
            "group_label": ["TD3", "TD3", "TD3", "TD3"],
            "seed_index": [0, 1, 2, 3],
            "total_training_env_steps": [10, 10, 10, 10],
            "mean_return_raw": [0.0, 1.0, 2.0, 100.0],
        }
    )

    figure = plot_experiment_group_learning_curves(
        group_metrics,
        seed_aggregate="interquartile_mean",
        n_bootstrap=200,
    )
    axis = figure.axes[0]

    assert axis.get_ylabel() == INTERQUARTILE_MEAN_Y_LABEL
    assert axis.lines[0].get_ydata().tolist() == [1.5]
    plt.close(figure)


def test_grid_plot_uses_fixed_layout_and_single_shared_legend() -> None:
    """The 2x2 grid orders panels by environment and storage capacity."""
    panels = [
        _grid_panel("Deterministic", "200"),
        _grid_panel("OU", "200"),
        _grid_panel("Deterministic", "30"),
        _grid_panel("OU", "30"),
    ]

    figure = plot_experiment_group_learning_curve_grid(
        panels,
        n_bootstrap=200,
    )
    axes = figure.axes
    legend_labels = [
        text.get_text() for text in figure.legends[0].get_texts()
    ]

    assert [axis.get_title() for axis in axes] == [
        "Deterministic, Speicherkapazität 200",
        "OU, Speicherkapazität 200",
        "Deterministic, Speicherkapazität 30",
        "OU, Speicherkapazität 30",
    ]
    assert all(axis.get_legend() is None for axis in axes)
    assert len(figure.legends) == 1
    assert legend_labels == ["PPO", "LSMC"]
    assert figure._supxlabel.get_text() == "Trainingsschritte"
    assert figure._supylabel.get_text() == DEFAULT_Y_LABEL
    plt.close(figure)


def test_grid_plot_shares_y_axis_by_row_by_default() -> None:
    """Grid plots use one y-axis scale per capacity row by default."""
    panels = [
        _grid_panel("Deterministic", "200", mean_returns=[210.0, 225.0]),
        _grid_panel("OU", "200", mean_returns=[205.0, 220.0]),
        _grid_panel("Deterministic", "30", mean_returns=[70.0, 85.0]),
        _grid_panel("OU", "30", mean_returns=[65.0, 80.0]),
    ]

    figure = plot_experiment_group_learning_curve_grid(
        panels,
        n_bootstrap=200,
    )
    top_left, top_right, bottom_left, bottom_right = figure.axes

    assert top_left.get_ylim() == top_right.get_ylim()
    assert bottom_left.get_ylim() == bottom_right.get_ylim()
    assert top_left.get_ylim() != bottom_left.get_ylim()
    plt.close(figure)


def test_grid_plot_requires_exactly_four_panels() -> None:
    """Grid plots validate their fixed 2x2 shape."""
    try:
        plot_experiment_group_learning_curve_grid([_grid_panel("OU", "30")])
    except ValueError as error:
        assert "Exactly four panels" in str(error)
    else:
        raise AssertionError("Expected ValueError for invalid grid panel count")


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


def _grid_panel(
    environment_label: str,
    capacity: str,
    *,
    mean_returns: list[float] | None = None,
) -> LearningCurveGridPanel:
    """Creates a small grid panel fixture."""
    returns = mean_returns or [3.0, 7.0]
    return LearningCurveGridPanel(
        environment_label=environment_label,
        capacity=capacity,
        group_metrics=pd.DataFrame(
            {
                "group_label": ["PPO", "PPO", "PPO", "PPO"],
                "seed_index": [0, 0, 1, 1],
                "total_training_env_steps": [0, 10, 0, 10],
                "mean_return_raw": [
                    returns[0],
                    returns[1],
                    returns[0],
                    returns[1],
                ],
            }
        ),
        benchmark_metrics=pd.DataFrame(
            {
                "method": ["lsmc", "lsmc"],
                "total_training_env_steps": [0, 10],
                "mean_return_raw": [4.0, 4.0],
            }
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Writes a small CSV fixture."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
