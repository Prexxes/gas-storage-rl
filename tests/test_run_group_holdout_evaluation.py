"""Tests for experiment-group holdout evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from gas_storage_rl.evaluation import run_group_holdout_evaluation


def test_group_holdout_logs_runs_episodes_references_and_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A group holdout run evaluates each member and aggregates over seeds."""
    group_dir = tmp_path / "runs" / "experiment_groups" / "group-1"
    run_0 = tmp_path / "runs" / "run-0"
    run_1 = tmp_path / "runs" / "run-1"
    group_dir.mkdir(parents=True)
    run_0.mkdir(parents=True)
    run_1.mkdir(parents=True)
    _write_json(
        group_dir / "group_config.json",
        {
            "seeds": {
                "dataset_seed": 20,
                "eval_seed": 50,
            },
        },
    )
    _write_csv(
        group_dir / "runs.csv",
        [
            _run_row(group_dir.name, 0, run_0),
            _run_row(group_dir.name, 1, run_1),
        ],
    )
    references = [
        {
            "split": "test",
            "path_id": 0,
            "episode_perfect_foresight_return_raw": 20.0,
        }
    ]
    calls = []

    def fake_references(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
        assert config["seeds"]["dataset_seed"] == 20
        assert split == "test"
        return references

    def fake_evaluate(
        run_dir: str | Path,
        split: str,
        *,
        model_checkpoint: str,
        write_final_episode_metrics: bool,
        perfect_foresight_references: list[dict[str, Any]],
        n_bootstrap: int,
        bootstrap_seed: int,
    ) -> dict[str, Any]:
        del n_bootstrap, bootstrap_seed
        run_path = Path(run_dir)
        seed_index = 0 if run_path == run_0 else 1
        calls.append((run_path, model_checkpoint, write_final_episode_metrics))
        assert split == "test"
        assert model_checkpoint == "best_validation"
        assert write_final_episode_metrics is True
        assert perfect_foresight_references == references
        mean_return_raw = 10.0 if seed_index == 0 else 30.0
        _write_csv(
            run_path / "final_episode_metrics_test.csv",
            [
                {
                    "split": "test",
                    "path_id": 0,
                    "episode_return_raw": mean_return_raw,
                    "episode_perfect_foresight_return_raw": 20.0,
                    "episode_perfect_foresight_ratio": mean_return_raw / 20.0,
                    "episode_optimality_gap": (20.0 - mean_return_raw) / 20.0,
                }
            ],
        )
        return {
            "mean_return_scaled": mean_return_raw / 10.0,
            "mean_return_raw": mean_return_raw,
            "median_return_raw": mean_return_raw,
            "std_return_raw": 0.0,
            "min_return_raw": mean_return_raw,
            "max_return_raw": mean_return_raw,
            "interquartile_mean_return_raw": mean_return_raw,
            "mean_terminal_deviation": 0.0,
            "mean_cumulative_cashflow": mean_return_raw,
            "mean_terminal_penalty": 0.0,
            "mean_number_of_constrained_actions": float(seed_index),
            "mean_number_of_physical_clipped_actions": float(seed_index + 1),
            "mean_number_of_terminal_feasibility_clipped_actions": 2.0,
            "mean_number_of_physical_only_clipped_actions": 1.0,
            "mean_number_of_terminal_only_clipped_actions": 0.0,
            "mean_number_of_physical_and_terminal_clipped_actions": 1.0,
            "mean_perfect_foresight_return_raw": 20.0,
            "mean_perfect_foresight_ratio": mean_return_raw / 20.0,
            "mean_optimality_gap": (20.0 - mean_return_raw) / 20.0,
        }

    monkeypatch.setattr(
        run_group_holdout_evaluation,
        "_build_group_references",
        fake_references,
    )
    monkeypatch.setattr(
        run_group_holdout_evaluation,
        "evaluate_holdout_run",
        fake_evaluate,
    )

    summary = run_group_holdout_evaluation.run_group_holdout_evaluation(
        group_dir,
        "test",
        n_bootstrap=100,
        bootstrap_seed=123,
    )

    assert calls == [
        (run_0, "best_validation", True),
        (run_1, "best_validation", True),
    ]
    assert summary["n_evaluated_runs"] == 2
    assert summary["mean_return_raw_over_seed"] == 20.0
    assert summary["interquartile_mean_return_raw_over_seed"] == 20.0
    assert (
        summary["metric_aggregates_over_seed"]["mean_perfect_foresight_ratio"][
            "mean"
        ]
        == 1.0
    )
    assert (group_dir / "perfect_foresight_references_test.jsonl").exists()
    assert (group_dir / "holdout_test_summary.json").exists()
    assert (group_dir / "holdout_test_runs.csv").exists()
    assert (group_dir / "holdout_test_episode_metrics.csv").exists()

    run_rows = _read_csv(group_dir / "holdout_test_runs.csv")
    assert [row["holdout_status"] for row in run_rows] == ["completed", "completed"]
    assert [row["model_checkpoint"] for row in run_rows] == [
        "best_validation",
        "best_validation",
    ]

    episode_rows = _read_csv(group_dir / "holdout_test_episode_metrics.csv")
    assert [row["seed_index"] for row in episode_rows] == ["0", "1"]
    assert {row["episode_perfect_foresight_return_raw"] for row in episode_rows} == {
        "20.0"
    }


def _run_row(group_id: str, seed_index: int, run_dir: Path) -> dict[str, Any]:
    """Returns a minimal experiment-group run row."""
    return {
        "experiment_group_id": group_id,
        "algorithm_name": "ppo",
        "seed_index": seed_index,
        "master_seed": 10,
        "dataset_seed": 20,
        "eval_seed": 50,
        "env_seed": 30 + seed_index,
        "agent_seed": 40 + seed_index,
        "status": "completed",
        "run_dir": str(run_dir),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes a JSON test fixture."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes CSV test rows."""
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Reads CSV test rows."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
