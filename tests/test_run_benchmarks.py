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
        "seeds": {"eval_seed": 1},
        "evaluation_config": {"lsmc_action_grid": [-1.0, 0.0, 1.0]},
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
        {"train": paths, "validation": paths + 1.0, "test": paths + 2.0},
        {"train": 1, "validation": 2, "test": 3},
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
