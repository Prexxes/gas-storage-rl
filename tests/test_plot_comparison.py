"""Tests for comparison plotting helpers."""

import pandas as pd

from gas_storage_rl.plotting.plot_comparison import (
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
