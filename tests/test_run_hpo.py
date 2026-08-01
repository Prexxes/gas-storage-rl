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


def _write_evaluations_csv(run_dir: Path, rows: list[dict]) -> None:
    """Writes minimal validation rows for HPO objective tests."""
    run_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (run_dir / "evaluations.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        assert config["environment_config"]["reward_scale"] == 2.0
        assert config["agent_config"]["ppo"]["device"] == "cpu"
        assert config["agent_config"]["ppo"]["n_steps"] == 1024
        assert config["agent_config"]["ppo"]["gamma"] == 1.0
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
    assert trial_rows[0]["reward_scale_multiplier"] == "1.0"
    assert trial_rows[0]["base_reward_scale"] == "2.0"
    assert trial_rows[0]["effective_reward_scale"] == "2.0"
    seed_rows = list(
        csv.DictReader((study_dir / "trial_seed_runs.csv").open(encoding="utf-8"))
    )
    assert [row["seed_index"] for row in seed_rows] == ["0", "1", "2"]
    assert {row["dataset_seed"] for row in seed_rows} == {"11"}
    assert {row["effective_reward_scale"] for row in seed_rows} == {"2.0"}
    assert {row["hpo_objective_selection"] for row in seed_rows} == {
        "final_mean_return_raw"
    }


def test_run_trial_uses_best_validation_row_for_hpo_objective(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """HPO selects max validation mean return instead of the final return."""
    config = _base_config(tmp_path)
    study_dir = tmp_path / "runs" / "hpo" / "study"
    study_dir.mkdir(parents=True)
    run_dir = tmp_path / "run-0"

    def fake_run_experiment(config, algorithm, **kwargs):
        del config, algorithm, kwargs
        _write_evaluations_csv(
            run_dir,
            [
                {
                    "total_training_env_steps": 0,
                    "mean_return_raw": 10.0,
                    "std_return_raw": 2.0,
                    "median_return_raw": 9.0,
                    "min_return_raw": 8.0,
                    "mean_terminal_deviation": 0.6,
                    "mean_number_of_constrained_actions": 3.0,
                },
                {
                    "total_training_env_steps": 8,
                    "mean_return_raw": 15.0,
                    "std_return_raw": 1.5,
                    "median_return_raw": 14.0,
                    "min_return_raw": 13.0,
                    "mean_terminal_deviation": 0.4,
                    "mean_number_of_constrained_actions": 2.0,
                },
                {
                    "total_training_env_steps": 16,
                    "mean_return_raw": 12.0,
                    "std_return_raw": 1.0,
                    "median_return_raw": 11.0,
                    "min_return_raw": 10.0,
                    "mean_terminal_deviation": 0.5,
                    "mean_number_of_constrained_actions": 1.0,
                },
            ],
        )
        return {
            "status": "completed",
            "run_dir": str(run_dir),
            "validation": {
                "mean_return_raw": 12.0,
                "std_return_raw": 1.0,
                "median_return_raw": 11.0,
                "min_return_raw": 10.0,
                "mean_terminal_deviation": 0.5,
                "mean_number_of_constrained_actions": 1.0,
            },
        }

    monkeypatch.setattr(run_hpo, "run_experiment", fake_run_experiment)

    objective = run_hpo._run_trial(
        FixedTrial(),
        config,
        "ppo",
        "study",
        [0],
        500_000,
        False,
        study_dir,
    )

    assert objective == 15.0
    run_hpo._export_trial_csvs(study_dir)
    seed_rows = list(
        csv.DictReader((study_dir / "trial_seed_runs.csv").open(encoding="utf-8"))
    )
    assert seed_rows[0]["hpo_objective_selection"] == "max_mean_return_raw"
    assert seed_rows[0]["hpo_objective_training_env_steps"] == "8.0"
    assert seed_rows[0]["hpo_objective_mean_validation_return_raw"] == "15.0"
    assert seed_rows[0]["mean_validation_return_raw"] == "15.0"
    assert seed_rows[0]["final_mean_validation_return_raw"] == "12.0"


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


def test_run_hpo_resumes_existing_study_to_target_trial_count(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Resume mode reuses an existing study and runs only missing trials."""
    config = _base_config(tmp_path)
    seen_trial_groups = []

    def fake_run_experiment(config, algorithm, **kwargs):
        del config, algorithm
        seen_trial_groups.append(kwargs["experiment_group_id"])
        return {
            "status": "completed",
            "run_dir": str(tmp_path / f"run-{len(seen_trial_groups)}"),
            "validation": {
                "mean_return_raw": float(len(seen_trial_groups)),
                "std_return_raw": 0.0,
                "median_return_raw": float(len(seen_trial_groups)),
                "min_return_raw": float(len(seen_trial_groups)),
                "mean_terminal_deviation": 0.0,
                "mean_number_of_constrained_actions": 0.0,
            },
        }

    monkeypatch.setattr(run_hpo, "run_experiment", fake_run_experiment)

    first_summary = run_hpo.run_hpo(
        config,
        "config",
        "ppo",
        n_trials=1,
        seed_indices=[0],
        total_timesteps=16,
        study_name="resume-study",
    )
    second_summary = run_hpo.run_hpo(
        config,
        "config",
        "ppo",
        n_trials=3,
        seed_indices=[0],
        total_timesteps=16,
        resume_study_dir=first_summary["study_dir"],
    )

    assert second_summary["study_dir"] == first_summary["study_dir"]
    assert second_summary["n_existing_finished_trials"] == 1
    assert second_summary["n_remaining_trials"] == 2
    trial_rows = list(
        csv.DictReader(
            (Path(first_summary["study_dir"]) / "trials.csv").open(
                encoding="utf-8",
            )
        )
    )
    assert len(trial_rows) == 3
    metadata = json.loads(
        (Path(first_summary["study_dir"]) / "metadata.json").read_text(
            encoding="utf-8",
        )
    )
    assert metadata["is_resume"] is True
    assert metadata["target_n_trials"] == 3


def test_validate_hpo_inputs_rejects_duplicate_seed_indices() -> None:
    """Seed indices must be disjoint within phase 1."""
    with pytest.raises(ValueError, match="seed_indices must be unique"):
        run_hpo._validate_hpo_inputs("ppo", 32, [0, 1, 1])


def test_agent_config_with_hyperparameters_preserves_base_runtime_settings(
    tmp_path: Path,
) -> None:
    """HPO keeps non-tuned base settings such as the SB3 device."""
    config = _base_config(tmp_path)

    merged = run_hpo._agent_config_with_hyperparameters(
        config,
        "ppo",
        {"learning_rate": 1e-4, "n_steps": 256},
    )

    assert merged["device"] == "cpu"
    assert merged["policy"] == "MlpPolicy"
    assert merged["learning_rate"] == 1e-4
    assert merged["n_steps"] == 256


def test_validate_hpo_inputs_rejects_nonpositive_n_jobs() -> None:
    """Trial-level parallelism requires a positive worker count."""
    with pytest.raises(ValueError, match="n_jobs must be positive"):
        run_hpo._validate_hpo_inputs("ppo", 32, [0, 1, 2], 0)


def test_validate_hpo_inputs_rejects_negative_startup_trials() -> None:
    """TPE startup trial count cannot be negative."""
    with pytest.raises(ValueError, match="n_startup_trials must be non-negative"):
        run_hpo._validate_hpo_inputs("ppo", 32, [0, 1, 2], 1, -1)
