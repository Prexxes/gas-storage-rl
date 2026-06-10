"""Tests for the oracle-cloned neural benchmark policy."""

from __future__ import annotations

import numpy as np

from gas_storage_rl.baselines.oracle_cloned_policy import OracleClonedPolicy


def test_oracle_cloned_policy_predicts_bounded_action() -> None:
    """Policy predictions have the expected SB3-compatible action shape."""
    policy = OracleClonedPolicy(observation_dim=6, hidden_sizes=(8,), seed=1)

    action, state = policy.predict(np.zeros(6, dtype=np.float32))

    assert state is None
    assert action.shape == (1,)
    assert action.dtype == np.float32
    assert -1.0 <= action[0] <= 1.0


def test_oracle_cloned_policy_fit_reduces_imitation_loss() -> None:
    """Supervised fitting reduces MSE on a simple deterministic target."""
    observations = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.5, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.5, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    actions = np.array([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
    policy = OracleClonedPolicy(observation_dim=6, hidden_sizes=(16,), seed=2)

    history = policy.fit(
        observations,
        actions,
        epochs=80,
        batch_size=4,
        learning_rate=5e-2,
        seed=2,
    )

    assert history[-1]["loss"] < history[0]["loss"]


def test_oracle_cloned_policy_is_deterministic_for_same_seed() -> None:
    """The same seed and data produce the same prediction."""
    observations = np.eye(6, dtype=np.float32)
    actions = np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(-1, 1)
    first = OracleClonedPolicy(observation_dim=6, hidden_sizes=(8,), seed=3)
    second = OracleClonedPolicy(observation_dim=6, hidden_sizes=(8,), seed=3)

    first.fit(
        observations,
        actions,
        epochs=3,
        batch_size=2,
        learning_rate=1e-2,
        seed=3,
    )
    second.fit(
        observations,
        actions,
        epochs=3,
        batch_size=2,
        learning_rate=1e-2,
        seed=3,
    )

    observation = np.ones(6, dtype=np.float32)
    assert np.allclose(first.predict(observation)[0], second.predict(observation)[0])


def test_oracle_cloned_policy_save_and_load(tmp_path) -> None:
    """Saved oracle-cloned policies can be loaded for diagnostics."""
    policy = OracleClonedPolicy(observation_dim=6, hidden_sizes=(8,), seed=4)
    observation = np.ones(6, dtype=np.float32)
    expected_action = policy.predict(observation)[0]

    path = policy.save(tmp_path / "oracle_cloned_policy.pt")
    loaded = OracleClonedPolicy.load(path)

    np.testing.assert_allclose(loaded.predict(observation)[0], expected_action)
