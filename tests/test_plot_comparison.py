"""Tests for comparison plotting helpers."""

import matplotlib.pyplot as plt
import pandas as pd

from gas_storage_rl.plotting.plot_comparison import (
    MEDIAN_COLOR,
    MEAN_COLOR,
    plot_final_return_violins,
    plot_regret_violins,
    relative_regret_to_perfect_foresight,
)


def test_relative_regret_can_exceed_one_for_negative_method_return() -> None:
    """Relative regret captures losses below zero when PF value is positive."""
    metrics = pd.DataFrame(
        [
            {
                "split": "test",
                "path_id": 0,
                "method": "perfect_foresight",
                "episode_return_raw": 2000.0,
            },
            {
                "split": "test",
                "path_id": 0,
                "method": "ppo",
                "episode_return_raw": -100.0,
            },
        ]
    )

    regret = relative_regret_to_perfect_foresight(metrics)

    assert regret.loc[0, "relative_regret_to_pf"] == 1.05


def test_violin_plot_marks_mean_and_median_with_legend() -> None:
    """Mean and median use distinct colors explained by a legend."""
    metrics = pd.DataFrame(
        {
            "method": ["ppo", "ppo", "ppo", "lsmc", "lsmc", "lsmc"],
            "episode_return_raw": [1.0, 2.0, 6.0, -1.0, 0.0, 4.0],
        }
    )

    figure = plot_final_return_violins(metrics)
    axis = figure.axes[0]
    legend = axis.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    colors = [handle.get_color() for handle in legend.legend_handles]

    assert labels == ["Mean", "Median"]
    assert colors == [MEAN_COLOR, MEDIAN_COLOR]
    plt.close(figure)


def test_final_return_violin_orders_benchmarks_before_rl_methods() -> None:
    """Final return violins use a stable benchmark-first method order."""
    metrics = pd.DataFrame(
        {
            "method": [
                "ppo",
                "perfect_foresight",
                "random",
                "td3",
                "lsmc",
                "oracle_cloned_policy",
                "rule_based",
                "sac",
            ],
            "episode_return_raw": [1.0, 8.0, 0.0, 2.0, 4.0, 5.0, 3.0, 6.0],
        }
    )

    figure = plot_final_return_violins(metrics)
    labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]

    assert labels == [
        "random",
        "rule_based",
        "lsmc",
        "oracle_cloned_policy",
        "perfect_foresight",
        "ppo",
        "td3",
        "sac",
    ]
    plt.close(figure)


def test_regret_violin_orders_benchmarks_before_rl_without_perfect_foresight() -> None:
    """Relative regret violins omit PF and keep benchmark-first ordering."""
    regret = pd.DataFrame(
        {
            "method": [
                "ppo",
                "random",
                "td3",
                "lsmc",
                "oracle_cloned_policy",
                "rule_based",
                "sac",
            ],
            "relative_regret_to_pf": [0.4, 1.0, 0.3, 0.2, 0.15, 0.5, 0.35],
        }
    )

    figure = plot_regret_violins(regret)
    labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]

    assert labels == [
        "random",
        "rule_based",
        "lsmc",
        "oracle_cloned_policy",
        "ppo",
        "td3",
        "sac",
    ]
    plt.close(figure)
