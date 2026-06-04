"""Tests for evaluation metrics."""

import numpy as np

from gas_storage_rl.evaluation.metrics import (
    aulc,
    interquartile_mean,
    summarize_episode_infos,
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


def test_episode_summary_uses_economic_reward_for_evaluation() -> None:
    """Episode return is economic cashflow plus terminal penalty."""
    summary = summarize_episode_infos(
        [
            {
                "raw_cashflow": -10.0,
                "terminal_penalty": 0.0,
                "economic_reward_raw": -10.0,
                "shaped_reward_raw": 5.0,
                "scaled_reward": 0.5,
                "reward_scale": 10.0,
                "executed_action": 1.0,
                "requested_action": 1.0,
                "storage_level": 1.0,
                "target_terminal_inventory": 0.0,
            },
            {
                "raw_cashflow": 20.0,
                "terminal_penalty": -1.0,
                "economic_reward_raw": 19.0,
                "shaped_reward_raw": 3.0,
                "scaled_reward": 0.3,
                "reward_scale": 10.0,
                "executed_action": -1.0,
                "requested_action": -1.0,
                "storage_level": 0.0,
                "target_terminal_inventory": 0.0,
            },
        ]
    )
    assert summary["episode_return_raw"] == 9.0
    assert summary["episode_return_scaled"] == 0.9
    assert summary["episode_shaped_return_raw"] == 8.0
    assert summary["episode_shaped_return_scaled"] == 0.8
    assert summary["cumulative_cashflow"] == 10.0
