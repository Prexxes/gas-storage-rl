"""Tests for training-run deduplication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gas_storage_rl.training.config import build_effective_run_config
from gas_storage_rl.training.run_experiment import (
    find_completed_run,
    run_config_fingerprint,
    run_experiment,
)


def _base_config(tmp_path: Path) -> dict:
    """Returns a minimal training config for rerun checks."""
    return {
        "environment_config": {
            "environment_name": "deterministic",
            "capacity": 30,
            "episode_length": 8,
        },
        "dataset_config": {
            "n_train_paths": 8,
            "n_validation_paths": 4,
            "n_test_paths": 4,
            "cache_dir": "data/cache",
            "use_cache": True,
            "force_regenerate": False,
        },
        "training_config": {
            "total_timesteps": 16,
            "eval_freq": 8,
            "n_seeds": 1,
        },
        "evaluation_config": {"deterministic": True},
        "seeds": {
            "master_seed": 1,
            "dataset_seed": 11,
            "env_seed": 12,
            "agent_seed": 13,
            "eval_seed": 14,
        },
        "price_process_config": {
            "seasonal_level": 2.0,
            "seasonal_amplitude": 1.0,
        },
        "agent_config": {
            "ppo": {
                "policy": "MlpPolicy",
                "device": "cpu",
                "gamma": 1.0,
            },
        },
        "logging_config": {"run_dir": str(tmp_path / "runs")},
    }


def _write_completed_run(
    run_dir: Path,
    effective_config: dict,
    summary: dict | None = None,
) -> None:
    """Writes the run files required for deduplication."""
    run_dir.mkdir(parents=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(effective_config, file)
    with (run_dir / "final_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary or {"algorithm_name": "ppo"}, file)


def test_run_fingerprint_ignores_group_id_and_logging_dir() -> None:
    """Organizational run metadata does not change rerun identity."""
    config = {
        "agent_config": {"algorithm_name": "ppo"},
        "seeds": {"agent_seed": 1},
        "logging_config": {"run_dir": "runs-a"},
        "experiment_group_id": "group-a",
    }
    same_run_config = {
        "agent_config": {"algorithm_name": "ppo"},
        "seeds": {"agent_seed": 1},
        "logging_config": {"run_dir": "runs-b"},
        "experiment_group_id": "group-b",
    }

    assert run_config_fingerprint(config) == run_config_fingerprint(same_run_config)


def test_find_completed_run_matches_config_without_group_id(tmp_path: Path) -> None:
    """Completed runs are found even when their group id differs."""
    config = _base_config(tmp_path)
    effective_config = build_effective_run_config(config, "ppo", seed_index=0)
    stored_config = dict(effective_config)
    stored_config["experiment_group_id"] = "old-group"
    run_dir = tmp_path / "runs" / "20260625-120000-deterministic-ppo-abc12345"
    _write_completed_run(run_dir, stored_config)

    assert find_completed_run(config["logging_config"]["run_dir"], effective_config) == run_dir


def test_run_experiment_skips_existing_completed_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A matching completed run prevents duplicate training by default."""
    config = _base_config(tmp_path)
    effective_config = build_effective_run_config(config, "ppo", seed_index=0)
    stored_config = dict(effective_config)
    stored_config["experiment_group_id"] = "old-group"
    run_dir = tmp_path / "runs" / "20260625-120000-deterministic-ppo-abc12345"
    _write_completed_run(
        run_dir,
        stored_config,
        {"status": "completed", "algorithm_name": "ppo", "seed_index": 0},
    )

    def fail_if_environment_is_built(*args, **kwargs):
        raise AssertionError("environment should not be built for skipped runs")

    monkeypatch.setattr(
        "gas_storage_rl.training.run_experiment.build_environment",
        fail_if_environment_is_built,
    )

    summary = run_experiment(
        config,
        "ppo",
        seed_index=0,
        experiment_group_id="new-group",
    )

    assert summary["status"] == "skipped"
    assert summary["run_dir"] == str(run_dir)
    assert summary["existing_run_dir"] == str(run_dir)


def test_run_experiment_rerun_bypasses_existing_completed_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The rerun flag forces training even when a matching run exists."""
    config = _base_config(tmp_path)
    effective_config = build_effective_run_config(config, "ppo", seed_index=0)
    run_dir = tmp_path / "runs" / "20260625-120000-deterministic-ppo-abc12345"
    _write_completed_run(run_dir, effective_config)

    def raise_after_skip_check(*args, **kwargs):
        raise RuntimeError("training path reached")

    monkeypatch.setattr(
        "gas_storage_rl.training.run_experiment.build_environment",
        raise_after_skip_check,
    )

    with pytest.raises(RuntimeError, match="training path reached"):
        run_experiment(config, "ppo", seed_index=0, rerun=True)


def test_run_experiment_logs_final_validation_after_exact_eval_step(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Final validation is appended even if periodic validation hit the last step."""
    config = _base_config(tmp_path)
    fake_loggers = []

    class FakeExperimentLogger:
        """Collects run output in memory."""

        def __init__(self, base_dir: str | Path, effective_config: dict) -> None:
            self.run_dir = Path(base_dir) / "fake-run"
            self.run_dir.mkdir(parents=True)
            self.rows: list[tuple[str, dict]] = []
            fake_loggers.append(self)

        def metadata(self) -> dict:
            """Returns minimal metadata."""
            return {}

        def write_json(self, name: str, payload: dict) -> None:
            """Ignores JSON writes."""

        def append_csv(self, name: str, row: dict) -> None:
            """Collects CSV writes."""
            self.rows.append((name, dict(row)))

        def finalize_metadata(self, metadata: dict) -> dict:
            """Returns metadata unchanged."""
            return metadata

    class FakeCallback:
        """Pretends periodic validation already ran at the final step."""

        def __init__(self, *args, **kwargs) -> None:
            self.last_validation_step = kwargs["total_timesteps"]

    class FakeModel:
        """Minimal SB3 model double."""

        def set_logger(self, logger) -> None:
            """Ignores SB3 logger configuration."""

        def learn(self, total_timesteps: int, callback) -> None:
            """Skips training."""

        def save(self, path: str | Path) -> None:
            """Ignores model persistence."""

    def fake_evaluate_policy_on_paths(*args, **kwargs):
        return (
            {
                "mean_return_raw": 10.0,
                "std_return_raw": 2.0,
                "split": "validation",
                "total_training_env_steps": kwargs["total_training_env_steps"],
            },
            [],
        )

    monkeypatch.setattr(
        "gas_storage_rl.training.run_experiment.build_environment",
        lambda config: (object(), object(), {}),
    )
    monkeypatch.setattr(
        "gas_storage_rl.training.run_experiment.GasStorageEnv",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "gas_storage_rl.training.run_experiment.ExperimentLogger",
        FakeExperimentLogger,
    )
    monkeypatch.setattr(
        "gas_storage_rl.training.run_experiment.make_sb3_agent",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        "gas_storage_rl.training.run_experiment.TrainingLoggingCallback",
        FakeCallback,
    )
    monkeypatch.setattr(
        "gas_storage_rl.training.run_experiment.evaluate_policy_on_paths",
        fake_evaluate_policy_on_paths,
    )
    monkeypatch.setattr(
        "gas_storage_rl.training.run_experiment._read_evaluation_rows",
        lambda run_dir: [
            {"total_training_env_steps": 0, "mean_return_raw": 0.0},
            {"total_training_env_steps": 8, "mean_return_raw": 8.0},
            {"total_training_env_steps": 16, "mean_return_raw": 16.0},
            {"total_training_env_steps": 16, "mean_return_raw": 10.0},
        ],
    )

    summary = run_experiment(config, "ppo", rerun=True)

    evaluation_rows = [
        row for name, row in fake_loggers[0].rows if name == "evaluations.csv"
    ]
    assert summary["validation"]["total_training_env_steps"] == 16
    assert summary["validation"]["AULC_validation_return_raw"] == 104.0
    assert summary["validation"]["max_validation_mean_return_raw"] == 10.0
    assert len(evaluation_rows) == 1
    assert evaluation_rows[0]["total_training_env_steps"] == 16
