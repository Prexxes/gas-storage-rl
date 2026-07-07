"""Tests for Stable-Baselines3 training callbacks."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from gas_storage_rl.training.callbacks import TrainingLoggingCallback


class _FakeLogger:
    """Collects callback CSV writes."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.rows: list[tuple[str, dict]] = []

    def append_csv(self, name: str, row: dict) -> None:
        """Stores a CSV row in memory."""
        self.rows.append((name, dict(row)))


class _FakeModel:
    """Collects model save paths."""

    def __init__(self) -> None:
        self.saved_paths: list[Path] = []
        self.logger = object()

    def save(self, path: str | Path) -> None:
        """Stores save destinations."""
        self.saved_paths.append(Path(path))


def test_validation_callback_saves_risk_adjusted_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Validation saves the best mean-return and risk-adjusted checkpoints."""
    logger = _FakeLogger(tmp_path)
    callback = TrainingLoggingCallback(
        experiment_logger=logger,
        eval_env=object(),
        eval_freq=1,
        algorithm_name="ppo",
        risk_adjusted_std_penalty=0.5,
    )
    callback.model = _FakeModel()
    callback.num_timesteps = 1

    def fake_evaluate_policy_on_paths(*args, **kwargs):
        return (
            {
                "mean_return_raw": 10.0,
                "std_return_raw": 4.0,
                "split": "validation",
                "total_training_env_steps": 1,
            },
            [],
        )

    monkeypatch.setattr(
        "gas_storage_rl.training.callbacks.evaluate_policy_on_paths",
        fake_evaluate_policy_on_paths,
    )

    callback._run_validation_if_due()

    assert logger.rows[0][1]["evaluation_phase"] == "callback"
    assert logger.rows[0][1]["risk_adjusted_return_raw"] == 8.0
    assert tmp_path / "best_validation_model" in callback.model.saved_paths
    assert (
        tmp_path / "best_risk_adjusted_validation_model"
        in callback.model.saved_paths
    )
