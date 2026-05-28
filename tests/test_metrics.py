"""Tests for evaluation metrics."""

import numpy as np

from gas_storage_rl.evaluation.metrics import (
    aulc,
    interquartile_mean,
    summarize_evaluation,
)


def test_metrics_aulc_and_interquartile_mean() -> None:
    """AULC and interquartile mean are computed."""
    assert aulc(np.array([0, 1, 2]), np.array([1, 2, 3])) == 4.0
    assert interquartile_mean(np.array([0, 1, 2, 100])) == 1.5


def test_evaluation_uses_total_training_env_steps() -> None:
    """Evaluation labels training progress explicitly."""
    metrics = summarize_evaluation(
        [
            {
                "episode_return_raw": 1.0,
                "episode_return_scaled": 0.1,
                "terminal_deviation": 0.0,
                "cumulative_cashflow": 1.0,
                "terminal_penalty": 0.0,
                "number_of_constrained_actions": 0.0,
            }
        ],
        split="validation",
        total_training_env_steps=50_000,
    )
    assert metrics["total_training_env_steps"] == 50_000
    assert "total_env_steps" not in metrics
