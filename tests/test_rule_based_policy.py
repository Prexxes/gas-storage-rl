"""Tests for rule-based policy."""

import numpy as np

from gas_storage_rl.baselines.rule_based_policy import RuleBasedPolicy


def test_rule_based_policy_feasibility() -> None:
    """Rule policy sells excess inventory near maturity."""
    policy = RuleBasedPolicy.from_training_prices(
        np.array([[10.0, 20.0, 30.0, 40.0]]),
        capacity=5.0,
        episode_length=4,
    )
    assert policy.act(storage_level=3.0, price=20.0, current_step=3) == -1.0
    assert policy.act(storage_level=0.0, price=100.0, current_step=1) == 0.0


def test_rule_based_policy_decodes_remaining_time() -> None:
    """Prediction derives the episode step from remaining time."""
    policy = RuleBasedPolicy.from_training_prices(
        np.array([[10.0, 20.0, 30.0, 40.0]]),
        capacity=5.0,
        episode_length=4,
    )
    observation = np.array([0.6, 0.4, 0.0, 1.0, 0.0], dtype=np.float32)

    action, _ = policy.predict(observation)

    assert action[0] == -1.0
