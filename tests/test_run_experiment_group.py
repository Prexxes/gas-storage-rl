"""Tests for sequential experiment groups."""

from __future__ import annotations

import csv
from pathlib import Path

from gas_storage_rl.training import run_experiment_group


def test_experiment_group_records_runs_and_continues_after_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """One failed seed is recorded without preventing later seeds from running."""
    config = {
        "logging_config": {"run_dir": str(tmp_path / "runs")},
        "training_config": {"n_seeds": 3},
        "seeds": {
            "master_seed": 10,
            "dataset_seed": 20,
            "env_seed": 30,
            "agent_seed": 40,
            "eval_seed": 50,
        },
    }
    called_indices = []

    def fake_run_experiment(config, algorithm, **kwargs):
        del algorithm
        seed_index = kwargs["seed_index"]
        called_indices.append(seed_index)
        if seed_index == 1:
            raise RuntimeError("intentional failure")
        return {"run_dir": str(tmp_path / f"run-{seed_index}")}

    monkeypatch.setattr(
        run_experiment_group,
        "run_experiment",
        fake_run_experiment,
    )

    summary = run_experiment_group.run_experiment_group(
        config,
        "debug",
        "ppo",
    )

    assert called_indices == [0, 1, 2]
    assert summary["completed_runs"] == 2
    assert summary["skipped_runs"] == 0
    assert summary["failed_runs"] == 1
    rows = list(
        csv.DictReader(
            (Path(summary["group_dir"]) / "runs.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert [row["status"] for row in rows] == ["completed", "failed", "completed"]
    assert {row["dataset_seed"] for row in rows} == {"20"}
    assert {row["eval_seed"] for row in rows} == {"50"}
    assert len({row["env_seed"] for row in rows}) == 3
    assert len({row["agent_seed"] for row in rows}) == 3


def test_experiment_group_records_skipped_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Skipped seed runs are reported separately from completed runs."""
    config = {
        "logging_config": {"run_dir": str(tmp_path / "runs")},
        "training_config": {"n_seeds": 2},
        "seeds": {
            "master_seed": 10,
            "dataset_seed": 20,
            "env_seed": 30,
            "agent_seed": 40,
            "eval_seed": 50,
        },
    }

    def fake_run_experiment(config, algorithm, **kwargs):
        del config, algorithm
        seed_index = kwargs["seed_index"]
        return {
            "status": "skipped" if seed_index == 0 else "completed",
            "run_dir": str(tmp_path / f"run-{seed_index}"),
            "validation": {
                "AULC_validation_return_raw": 100.0 + seed_index,
                "normalized_AULC_validation_return_raw": 10.0 + seed_index,
            },
        }

    monkeypatch.setattr(
        run_experiment_group,
        "run_experiment",
        fake_run_experiment,
    )

    summary = run_experiment_group.run_experiment_group(
        config,
        "debug",
        "ppo",
    )

    assert summary["completed_runs"] == 1
    assert summary["skipped_runs"] == 1
    assert summary["failed_runs"] == 0
    rows = list(
        csv.DictReader(
            (Path(summary["group_dir"]) / "runs.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert [row["status"] for row in rows] == ["skipped", "completed"]
    assert [row["AULC_validation_return_raw"] for row in rows] == ["100.0", "101.0"]
    assert [row["normalized_AULC_validation_return_raw"] for row in rows] == [
        "10.0",
        "11.0",
    ]


def test_experiment_group_accepts_explicit_seed_indices(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Manual final runs can use disjoint seed indices such as 100..107."""
    config = {
        "logging_config": {"run_dir": str(tmp_path / "runs")},
        "training_config": {"n_seeds": 2},
        "seeds": {
            "master_seed": 10,
            "dataset_seed": 20,
            "env_seed": 30,
            "agent_seed": 40,
            "eval_seed": 50,
        },
    }
    called_indices = []

    def fake_run_experiment(config, algorithm, **kwargs):
        del config, algorithm
        seed_index = kwargs["seed_index"]
        called_indices.append(seed_index)
        return {"status": "completed", "run_dir": str(tmp_path / f"run-{seed_index}")}

    monkeypatch.setattr(
        run_experiment_group,
        "run_experiment",
        fake_run_experiment,
    )

    summary = run_experiment_group.run_experiment_group(
        config,
        "debug",
        "ppo",
        seed_indices=[100, 101],
    )

    assert called_indices == [100, 101]
    assert summary["n_seeds"] == 2
    assert summary["seed_indices"] == [100, 101]
    rows = list(
        csv.DictReader(
            (Path(summary["group_dir"]) / "runs.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert [row["seed_index"] for row in rows] == ["100", "101"]


def test_experiment_group_rejects_duplicate_explicit_seed_indices(
    tmp_path: Path,
) -> None:
    """Explicit seed-index lists must be disjoint."""
    config = {
        "logging_config": {"run_dir": str(tmp_path / "runs")},
        "training_config": {"n_seeds": 2},
        "seeds": {
            "master_seed": 10,
            "dataset_seed": 20,
            "env_seed": 30,
            "agent_seed": 40,
            "eval_seed": 50,
        },
    }

    try:
        run_experiment_group.run_experiment_group(
            config,
            "debug",
            "ppo",
            seed_indices=[100, 100],
        )
    except ValueError as error:
        assert "seed_indices must be unique" in str(error)
    else:
        raise AssertionError("duplicate seed indices should fail")
