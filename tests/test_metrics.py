"""Tests for evaluation metrics."""

import numpy as np

from gas_storage_rl.evaluation.metrics import (
    add_risk_adjusted_return,
    aulc,
    interquartile_mean,
    summarize_episode_infos,
    summarize_evaluation,
    validation_return_aulc,
)


def test_metrics_aulc_and_interquartile_mean() -> None:
    """AULC and interquartile mean are computed."""
    assert aulc(np.array([0, 1, 2]), np.array([1, 2, 3])) == 4.0
    assert interquartile_mean(np.array([0, 1, 2, 100])) == 1.5


def test_validation_return_aulc_uses_last_duplicate_step() -> None:
    """Final validation replaces callback validation at the same step."""
    metrics = validation_return_aulc(
        [
            {"total_training_env_steps": 0, "mean_return_raw": 0.0},
            {"total_training_env_steps": 10, "mean_return_raw": 10.0},
            {"total_training_env_steps": 10, "mean_return_raw": 20.0},
        ],
        total_timesteps=10,
    )

    assert metrics["AULC_validation_return_raw"] == 100.0
    assert metrics["normalized_AULC_validation_return_raw"] == 10.0


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


def test_add_risk_adjusted_return_uses_mean_minus_std_penalty() -> None:
    """Risk-adjusted validation return penalizes path-wise volatility."""
    metrics = {"mean_return_raw": 12.0, "std_return_raw": 4.0}

    add_risk_adjusted_return(metrics, std_penalty=0.5)

    assert metrics["risk_adjusted_return_raw"] == 10.0
    assert metrics["risk_adjusted_std_penalty"] == 0.5


def test_episode_summary_uses_cashflow_reward_for_evaluation() -> None:
    """Episode return is cashflow plus terminal penalty."""
    summary = summarize_episode_infos(
        [
            {
                "raw_cashflow": -10.0,
                "terminal_penalty": 0.0,
                "raw_reward": -10.0,
                "scaled_reward": -1.0,
                "reward_scale": 10.0,
                "executed_action": 1.0,
                "requested_action": 1.0,
                "storage_level": 1.0,
                "target_terminal_inventory": 0.0,
            },
            {
                "raw_cashflow": 20.0,
                "terminal_penalty": -1.0,
                "raw_reward": 19.0,
                "scaled_reward": 1.9,
                "reward_scale": 10.0,
                "executed_action": -1.0,
                "requested_action": -1.0,
                "storage_level": 0.0,
                "target_terminal_inventory": 0.0,
            },
        ]
    )
    assert summary["episode_return_raw"] == 9.0
    assert np.isclose(summary["episode_return_scaled"], 0.9)
    assert summary["cumulative_cashflow"] == 10.0


def test_episode_summary_splits_physical_and_terminal_clipping() -> None:
    """Clipped actions are counted by physical and terminal-feasibility cause."""
    summary = summarize_episode_infos(
        [
            {
                "raw_cashflow": 0.0,
                "terminal_penalty": 0.0,
                "scaled_reward": 0.0,
                "executed_action": 0.5,
                "requested_action": 1.0,
                "rate_capacity_clipped_action": 0.5,
                "terminal_feasibility_clipped": False,
                "storage_level": 0.5,
                "target_terminal_inventory": 0.0,
            },
            {
                "raw_cashflow": 0.0,
                "terminal_penalty": 0.0,
                "scaled_reward": 0.0,
                "executed_action": 0.0,
                "requested_action": 0.5,
                "rate_capacity_clipped_action": 0.5,
                "terminal_feasibility_clipped": True,
                "storage_level": 0.5,
                "target_terminal_inventory": 0.0,
            },
            {
                "raw_cashflow": 0.0,
                "terminal_penalty": 0.0,
                "scaled_reward": 0.0,
                "executed_action": 0.0,
                "requested_action": 1.0,
                "rate_capacity_clipped_action": 0.5,
                "terminal_feasibility_clipped": True,
                "storage_level": 0.5,
                "target_terminal_inventory": 0.0,
            },
        ]
    )

    assert summary["number_of_constrained_actions"] == 3.0
    assert summary["number_of_physical_clipped_actions"] == 2.0
    assert summary["number_of_terminal_feasibility_clipped_actions"] == 2.0
    assert summary["number_of_physical_only_clipped_actions"] == 1.0
    assert summary["number_of_terminal_only_clipped_actions"] == 1.0
    assert summary["number_of_physical_and_terminal_clipped_actions"] == 1.0
