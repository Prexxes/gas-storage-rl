"""Tests for benchmark runner logging."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.storage_dynamics import StorageParams
from gas_storage_rl.evaluation import run_benchmarks


def _config(tmp_path: Path) -> dict[str, Any]:
    """Returns a minimal benchmark config."""
    return {
        "logging_config": {"run_dir": str(tmp_path / "runs")},
        "seeds": {"agent_seed": 2, "eval_seed": 1},
        "evaluation_config": {
            "lsmc_action_grid": [-1.0, 0.0, 1.0],
            "oracle_cloned_policy_epochs": 2,
            "oracle_cloned_policy_batch_size": 4,
            "oracle_cloned_policy_hidden_sizes": [8],
        },
    }


def _environment() -> tuple[PathDataset, StorageParams, dict[str, Any]]:
    """Returns a tiny deterministic benchmark environment."""
    paths = np.array(
        [
            [10.0, 20.0, 30.0],
            [30.0, 20.0, 10.0],
        ],
        dtype=np.float32,
    )
    dataset = PathDataset(
        {
            "pretrain": paths + 3.0,
            "train": paths,
            "validation": paths + 1.0,
            "test": paths + 2.0,
        },
        {"pretrain": 4, "train": 1, "validation": 2, "test": 3},
        {
            "pretrain": [
                {"start_date": "2023-01-01", "end_date": "2023-01-03"},
                {"start_date": "2023-01-04", "end_date": "2023-01-06"},
            ],
            "train": [
                {"start_date": "2024-01-01", "end_date": "2024-01-03"},
                {"start_date": "2024-01-04", "end_date": "2024-01-06"},
            ],
            "validation": [
                {"start_date": "2024-02-01", "end_date": "2024-02-03"},
                {"start_date": "2024-02-04", "end_date": "2024-02-06"},
            ],
            "test": [
                {"start_date": "2024-03-01", "end_date": "2024-03-03"},
                {"start_date": "2024-03-04", "end_date": "2024-03-06"},
            ],
        },
    )
    return dataset, StorageParams(capacity=2.0), {"seed": 1}


def test_main_logs_train_and_validation_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Benchmark CLI defaults to train and validation logging only."""
    config = _config(tmp_path)
    monkeypatch.setattr(run_benchmarks, "load_config", lambda _: config)
    monkeypatch.setattr(run_benchmarks, "build_environment", lambda _: _environment())
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmarks", "--config", "configs/debug.yaml"],
    )

    run_benchmarks.main()

    benchmark_dirs = list((tmp_path / "runs" / "benchmarks").iterdir())
    assert len(benchmark_dirs) == 1
    run_dir = benchmark_dirs[0]
    assert (run_dir / "benchmark_metrics_train.json").exists()
    assert (run_dir / "benchmark_metrics_validation.json").exists()
    assert not (run_dir / "benchmark_metrics_test.json").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["splits"] == ["train", "validation"]
    assert metadata["contains_test_split"] is False

    rows = list(
        csv.DictReader(
            (run_dir / "benchmark_metrics.csv").open(newline="", encoding="utf-8")
        )
    )
    assert len(rows) == 8
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert {row["benchmark"] for row in rows} == {
        "random",
        "rule_based",
        "lsmc",
        "perfect_foresight",
    }


def test_main_logs_test_only_when_explicit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Benchmark CLI evaluates test only when test is requested."""
    config = _config(tmp_path)
    monkeypatch.setattr(run_benchmarks, "load_config", lambda _: config)
    monkeypatch.setattr(run_benchmarks, "build_environment", lambda _: _environment())
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmarks", "--config", "configs/debug.yaml", "--split", "test"],
    )

    run_benchmarks.main()

    run_dir = next((tmp_path / "runs" / "benchmarks").iterdir())
    assert (run_dir / "benchmark_metrics_test.json").exists()
    assert not (run_dir / "benchmark_metrics_train.json").exists()
    assert not (run_dir / "benchmark_metrics_validation.json").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["splits"] == ["test"]
    assert metadata["contains_test_split"] is True

    rows = list(
        csv.DictReader(
            (run_dir / "benchmark_metrics.csv").open(newline="", encoding="utf-8")
        )
    )
    assert len(rows) == 4
    assert {row["split"] for row in rows} == {"test"}


def test_main_can_log_perfect_foresight_trajectories(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Perfect-foresight trajectories are logged as JSONL when requested."""
    config = _config(tmp_path)
    monkeypatch.setattr(run_benchmarks, "load_config", lambda _: config)
    monkeypatch.setattr(run_benchmarks, "build_environment", lambda _: _environment())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmarks",
            "--config",
            "configs/debug.yaml",
            "--split",
            "validation",
            "--write-perfect-foresight-trajectories",
        ],
    )

    run_benchmarks.main()

    run_dir = next((tmp_path / "runs" / "benchmarks").iterdir())
    trajectory_path = run_dir / "perfect_foresight_trajectories_validation.jsonl"
    assert trajectory_path.exists()
    rows = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["split"] == "validation"
    assert rows[0]["path_id"] == 0
    assert rows[0]["start_date"] == "2024-02-01"
    assert rows[0]["end_date"] == "2024-02-03"
    assert rows[0]["prices"] == [11.0, 21.0, 31.0]
    assert rows[0]["initial_inventory"] == 0.0
    assert rows[0]["target_inventory"] == 0.0
    assert len(rows[0]["actions"]) == 3
    assert len(rows[0]["storage_levels"]) == 4
    assert "objective_value" in rows[0]
    assert "terminal_deviation" in rows[0]
    assert rows[0]["success"] is True

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["writes_perfect_foresight_trajectories"] is True


def test_main_can_log_pretrain_split(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Benchmark CLI can evaluate the explicit pretrain split."""
    config = _config(tmp_path)
    monkeypatch.setattr(run_benchmarks, "load_config", lambda _: config)
    monkeypatch.setattr(run_benchmarks, "build_environment", lambda _: _environment())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmarks",
            "--config",
            "configs/debug.yaml",
            "--split",
            "pretrain",
            "--write-perfect-foresight-trajectories",
        ],
    )

    run_benchmarks.main()

    run_dir = next((tmp_path / "runs" / "benchmarks").iterdir())
    assert (run_dir / "benchmark_metrics_pretrain.json").exists()
    trajectory_path = run_dir / "perfect_foresight_trajectories_pretrain.jsonl"
    rows = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["split"] == "pretrain"
    assert rows[0]["start_date"] == "2023-01-01"


def test_main_can_evaluate_oracle_cloned_policy_on_validation_and_test(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Oracle-cloned policy trains on pretrain/train and reports validation/test."""
    config = _config(tmp_path)
    monkeypatch.setattr(run_benchmarks, "load_config", lambda _: config)
    monkeypatch.setattr(run_benchmarks, "build_environment", lambda _: _environment())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmarks",
            "--config",
            "configs/debug.yaml",
            "--split",
            "train",
            "--split",
            "validation",
            "--split",
            "test",
            "--include-oracle-cloned-policy",
        ],
    )

    run_benchmarks.main()

    run_dir = next((tmp_path / "runs" / "benchmarks").iterdir())
    train_metrics = json.loads(
        (run_dir / "benchmark_metrics_train.json").read_text(encoding="utf-8")
    )
    validation_metrics = json.loads(
        (run_dir / "benchmark_metrics_validation.json").read_text(encoding="utf-8")
    )
    test_metrics = json.loads(
        (run_dir / "benchmark_metrics_test.json").read_text(encoding="utf-8")
    )

    assert "oracle_cloned_policy" not in train_metrics["benchmarks"]
    assert "oracle_cloned_policy" in validation_metrics["benchmarks"]
    assert "oracle_cloned_policy" in test_metrics["benchmarks"]
    assert (
        validation_metrics["benchmarks"]["oracle_cloned_policy"][
            "imitation_training_samples"
        ]
        == 12.0
    )
    assert (
        test_metrics["benchmarks"]["oracle_cloned_policy"][
            "imitation_training_samples"
        ]
        == 12.0
    )

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["includes_oracle_cloned_policy"] is True

    rows = list(
        csv.DictReader(
            (run_dir / "benchmark_metrics.csv").open(newline="", encoding="utf-8")
        )
    )
    assert len(rows) == 14
    assert {
        row["split"]
        for row in rows
        if row["benchmark"] == "oracle_cloned_policy"
    } == {"validation", "test"}


def test_main_writes_benchmark_evaluations_on_timeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Benchmark evaluations are expanded over RL evaluation step points."""
    config = _config(tmp_path)
    config["training_config"] = {"total_timesteps": 50_000, "eval_freq": 20_000}
    monkeypatch.setattr(run_benchmarks, "load_config", lambda _: config)
    monkeypatch.setattr(run_benchmarks, "build_environment", lambda _: _environment())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmarks",
            "--config",
            "configs/debug.yaml",
            "--split",
            "validation",
        ],
    )

    run_benchmarks.main()

    run_dir = next((tmp_path / "runs" / "benchmarks").iterdir())
    rows = list(
        csv.DictReader(
            (run_dir / "benchmark_evaluations.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert {int(row["total_training_env_steps"]) for row in rows} == {
        0,
        20_000,
        40_000,
        50_000,
    }
    assert {row["method"] for row in rows} == {
        "random",
        "rule_based",
        "lsmc",
        "perfect_foresight",
    }
    assert {row["split"] for row in rows} == {"validation"}
    assert all(row["is_baseline_reference"] == "True" for row in rows)


def test_main_writes_final_episode_metrics_when_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Benchmark CLI writes one final episode metrics CSV per requested split."""
    config = _config(tmp_path)
    monkeypatch.setattr(run_benchmarks, "load_config", lambda _: config)
    monkeypatch.setattr(run_benchmarks, "build_environment", lambda _: _environment())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmarks",
            "--config",
            "configs/debug.yaml",
            "--split",
            "validation",
            "--write-final-episode-metrics",
        ],
    )

    run_benchmarks.main()

    run_dir = next((tmp_path / "runs" / "benchmarks").iterdir())
    rows = list(
        csv.DictReader(
            (run_dir / "final_episode_metrics_validation.csv").open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert len(rows) == 8
    assert {row["method"] for row in rows} == {
        "random",
        "rule_based",
        "lsmc",
        "perfect_foresight",
    }
    assert {row["path_id"] for row in rows} == {"0", "1"}
    assert {row["start_date"] for row in rows} == {
        "2024-02-01",
        "2024-02-04",
    }
    assert "episode_return_raw" in rows[0]

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["writes_final_episode_metrics"] is True
