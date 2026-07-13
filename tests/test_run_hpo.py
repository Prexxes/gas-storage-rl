"""Tests for phase-1 HPO orchestration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gas_storage_rl.hpo import run_hpo


class FixedTrial:
    """Trial stub with deterministic search-space choices."""

    number = 7

    def suggest_categorical(self, name, choices):
        """Returns deterministic categorical values."""
        del name
        return choices[0]

    def suggest_float(self, name, low, high, log=False):
        """Returns deterministic float values."""
        del name, log
        return (float(low) + float(high)) / 2.0


def _base_config(tmp_path: Path) -> dict:
    """Returns a minimal HPO config."""
    return {
        "environment_config": {
            "environment_name": "deterministic",
            "capacity": 30,
            "episode_length": 8,
            "reward_scale": 2.0,
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
            },
        },
        "logging_config": {"run_dir": str(tmp_path / "runs")},
    }


def test_run_trial_aggregates_three_seed_runs_and_writes_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A valid HPO trial requires all configured seed runs."""
    config = _base_config(tmp_path)
    study_dir = tmp_path / "runs" / "hpo" / "study"
    study_dir.mkdir(parents=True)
    called_seed_indices = []

    def fake_run_experiment(config, algorithm, **kwargs):
        del algorithm
        seed_index = kwargs["seed_index"]
        called_seed_indices.append(seed_index)
        assert config["training_config"]["total_timesteps"] == 500_000
        assert config["environment_config"]["reward_scale"] == 0.5
        return {
            "status": "completed",
            "run_dir": str(tmp_path / f"run-{seed_index}"),
            "validation": {
                "mean_return_raw": float(seed_index + 10),
                "std_return_raw": float(seed_index),
                "median_return_raw": float(seed_index + 9),
                "min_return_raw": float(seed_index + 8),
                "mean_terminal_deviation": 0.5,
                "mean_number_of_constrained_actions": 2.0,
            },
        }

    monkeypatch.setattr(run_hpo, "run_experiment", fake_run_experiment)

    objective = run_hpo._run_trial(
        FixedTrial(),
        config,
        "ppo",
        "study",
        [0, 1, 2],
        500_000,
        False,
        study_dir,
    )

    assert objective == 11.0
    assert called_seed_indices == [0, 1, 2]
    trial_payload = json.loads(
        (study_dir / "trial_0007.json").read_text(encoding="utf-8")
    )
    assert trial_payload["objective_mean_validation_return_raw"] == 11.0
    assert len(trial_payload["seed_runs"]) == 3
    run_hpo._export_trial_csvs(study_dir)
    trial_rows = list(
        csv.DictReader((study_dir / "trials.csv").open(encoding="utf-8"))
    )
    assert trial_rows[0]["objective_mean_validation_return_raw"] == "11.0"
    assert trial_rows[0]["median_validation_return_raw_across_seeds"] == "11.0"
    assert trial_rows[0]["reward_scale_multiplier"] == "0.25"
    assert trial_rows[0]["base_reward_scale"] == "2.0"
    assert trial_rows[0]["effective_reward_scale"] == "0.5"
    seed_rows = list(
        csv.DictReader((study_dir / "trial_seed_runs.csv").open(encoding="utf-8"))
    )
    assert [row["seed_index"] for row in seed_rows] == ["0", "1", "2"]
    assert {row["dataset_seed"] for row in seed_rows} == {"11"}
    assert {row["effective_reward_scale"] for row in seed_rows} == {"0.5"}


def test_run_trial_fails_when_one_seed_run_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A trial is invalid when any configured seed run fails."""
    config = _base_config(tmp_path)
    study_dir = tmp_path / "runs" / "hpo" / "study"
    study_dir.mkdir(parents=True)

    def fake_run_experiment(config, algorithm, **kwargs):
        del config, algorithm, kwargs
        raise RuntimeError("training failed")

    monkeypatch.setattr(run_hpo, "run_experiment", fake_run_experiment)

    with pytest.raises(RuntimeError, match="failed before all seed runs"):
        run_hpo._run_trial(
            FixedTrial(),
            config,
            "ppo",
            "study",
            [0, 1, 2],
            500_000,
            False,
            study_dir,
        )

    run_hpo._export_trial_csvs(study_dir)
    trial_rows = list(
        csv.DictReader((study_dir / "trials.csv").open(encoding="utf-8"))
    )
    assert trial_rows[0]["status"] == "failed"


def test_validate_hpo_inputs_rejects_duplicate_seed_indices() -> None:
    """Seed indices must be disjoint within phase 1."""
    with pytest.raises(ValueError, match="seed_indices must be unique"):
        run_hpo._validate_hpo_inputs("ppo", 32, [0, 1, 1])


def test_validate_hpo_inputs_rejects_nonpositive_n_jobs() -> None:
    """Trial-level parallelism requires a positive worker count."""
    with pytest.raises(ValueError, match="n_jobs must be positive"):
        run_hpo._validate_hpo_inputs("ppo", 32, [0, 1, 2], 0)
