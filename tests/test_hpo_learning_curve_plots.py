"""Tests for HPO learning curve plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gas_storage_rl.plotting.plot_hpo_learning_curves import (
    COMPARISON_LABEL,
    aggregate_learning_curves,
    plot_hpo_learning_curves,
    rank_trial_ids,
)
from gas_storage_rl.plotting.statistics import bootstrap_percentile_ci


def test_bootstrap_percentile_ci_contains_sample_mean() -> None:
    """Bootstrap intervals are deterministic and centered on plausible means."""
    lower, upper = bootstrap_percentile_ci(
        np.array([1.0, 2.0, 3.0]),
        n_bootstrap=500,
        random_seed=7,
    )

    assert lower <= 2.0 <= upper


def test_bootstrap_percentile_ci_single_sample_is_degenerate() -> None:
    """One seed yields a zero-width confidence interval."""
    lower, upper = bootstrap_percentile_ci(np.array([4.0]))

    assert (lower, upper) == (4.0, 4.0)


def test_hpo_learning_curves_aggregate_seeds_by_trial_and_step() -> None:
    """Each trial curve is the stepwise mean across seed evaluations."""
    metrics = pd.DataFrame(
        {
            "trial_id": [0, 0, 0, 0, 1, 1],
            "seed_index": [0, 0, 1, 1, 0, 0],
            "total_training_env_steps": [0, 10, 0, 10, 0, 10],
            "mean_return_raw": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0],
        }
    )

    aggregate = aggregate_learning_curves(metrics)
    trial_zero = aggregate[aggregate["trial_id"] == 0]

    assert trial_zero["mean_return_raw"].tolist() == [3.0, 5.0]


def test_hpo_ranking_prefers_objective_values() -> None:
    """Highlighted trials follow the HPO objective ranking when available."""
    aggregate = pd.DataFrame(
        {
            "trial_id": [0, 1],
            "total_training_env_steps": [10, 10],
            "mean_return_raw": [100.0, 1.0],
        }
    )
    ranking = pd.DataFrame(
        {
            "trial_id": [0, 1],
            "objective_mean_validation_return_raw": [10.0, 20.0],
        }
    )

    assert rank_trial_ids(aggregate, ranking) == [1, 0]


def test_hpo_overview_labels_only_top_three_and_sb3_default() -> None:
    """Overview keeps the legend compact and labels the comparison group."""
    trial_metrics = _trial_metrics(n_trials=4)
    comparison_metrics = _comparison_metrics()
    ranking = pd.DataFrame(
        {
            "trial_id": [0, 1, 2, 3],
            "objective_mean_validation_return_raw": [40.0, 30.0, 20.0, 10.0],
        }
    )

    figure = plot_hpo_learning_curves(
        trial_metrics,
        mode="overview",
        comparison_seed_metrics=comparison_metrics,
        ranking=ranking,
    )
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]

    assert labels == [
        "Top 1: Trial 0",
        "Top 2: Trial 1",
        "Top 3: Trial 2",
        "HPO trials",
        COMPARISON_LABEL,
    ]
    plt.close(figure)


def test_hpo_best_only_adds_bootstrap_bands_for_best_and_comparison() -> None:
    """Best-only mode plots confidence bands around seed-aggregated curves."""
    trial_metrics = _trial_metrics(n_trials=2)
    comparison_metrics = _comparison_metrics()
    ranking = pd.DataFrame(
        {
            "trial_id": [0, 1],
            "objective_mean_validation_return_raw": [10.0, 20.0],
        }
    )

    figure = plot_hpo_learning_curves(
        trial_metrics,
        mode="best_only",
        comparison_seed_metrics=comparison_metrics,
        ranking=ranking,
        n_bootstrap=200,
    )
    axis = figure.axes[0]
    labels = [text.get_text() for text in axis.get_legend().get_texts()]

    assert labels == ["Best HPO trial 1", COMPARISON_LABEL]
    assert len(axis.collections) == 2
    plt.close(figure)


def test_hpo_learning_curve_accepts_optional_title() -> None:
    """Optional titles are applied to the HPO learning curve axis."""
    figure = plot_hpo_learning_curves(
        _trial_metrics(n_trials=1),
        title="SAC HPO learning curves",
    )

    assert figure.axes[0].get_title() == "SAC HPO learning curves"
    plt.close(figure)


def _trial_metrics(n_trials: int) -> pd.DataFrame:
    """Returns synthetic per-seed HPO evaluation metrics."""
    rows = []
    for trial_id in range(n_trials):
        for seed_index in range(3):
            for step in [0, 10]:
                rows.append(
                    {
                        "trial_id": trial_id,
                        "seed_index": seed_index,
                        "total_training_env_steps": step,
                        "mean_return_raw": float(
                            100 - trial_id * 10 + seed_index + step
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _comparison_metrics() -> pd.DataFrame:
    """Returns synthetic per-seed comparison-group metrics."""
    rows = []
    for seed_index in range(3):
        for step in [0, 10]:
            rows.append(
                {
                    "seed_index": seed_index,
                    "total_training_env_steps": step,
                    "mean_return_raw": float(50 + seed_index + step),
                }
            )
    return pd.DataFrame(rows)
