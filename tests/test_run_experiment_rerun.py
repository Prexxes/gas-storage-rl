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
