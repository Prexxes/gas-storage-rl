"""Tests for paired reward ablation comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from gas_storage_rl.evaluation.compare_reward_ablation import (
    compare_reward_ablation,
    write_comparison_outputs,
)


def test_compare_reward_ablation_pairs_completed_seed_indices(tmp_path: Path) -> None:
    """Comparison reports paired metric deltas by seed index."""
    old_csv = _write_group(tmp_path, "old", {0: 10.0, 1: 20.0})
    new_csv = _write_group(tmp_path, "new", {0: 12.0, 1: 19.0})

    comparison = compare_reward_ablation(old_csv, new_csv)

    assert comparison["n_pairs"] == 2
    assert comparison["paired_seed_indices"] == [0, 1]
    assert comparison["metrics"]["mean_return_raw"] == {
        "mean_delta": 0.5,
        "min_delta": -1.0,
        "max_delta": 2.0,
        "positive_fraction": 0.5,
    }
    assert comparison["rows"][0]["delta_mean_return_raw"] == 2.0
    assert comparison["rows"][1]["delta_mean_return_raw"] == -1.0


def test_write_comparison_outputs_writes_csv_and_summary(tmp_path: Path) -> None:
    """Comparison outputs can be inspected as CSV and JSON files."""
    old_csv = _write_group(tmp_path, "old", {0: 10.0})
    new_csv = _write_group(tmp_path, "new", {0: 12.0})
    comparison = compare_reward_ablation(old_csv, new_csv)

    outputs = write_comparison_outputs(comparison, tmp_path / "comparison")

    assert Path(outputs["rows_csv"]).exists()
    assert Path(outputs["summary_json"]).exists()
    with Path(outputs["summary_json"]).open("r", encoding="utf-8") as file:
        summary = json.load(file)
    assert "rows" not in summary
    assert summary["n_pairs"] == 1


def test_compare_reward_ablation_accepts_skipped_existing_runs(
    tmp_path: Path,
) -> None:
    """Skipped group rows still point to reusable completed run summaries."""
    old_csv = _write_group(tmp_path, "old", {0: 10.0}, status="skipped")
    new_csv = _write_group(tmp_path, "new", {0: 12.0}, status="skipped")

    comparison = compare_reward_ablation(old_csv, new_csv)

    assert comparison["n_pairs"] == 1
    assert comparison["rows"][0]["delta_mean_return_raw"] == 2.0


def _write_group(
    tmp_path: Path,
    name: str,
    returns: dict[int, float],
    status: str = "completed",
) -> Path:
    """Writes a minimal experiment group and run summaries."""
    group_dir = tmp_path / name
    group_dir.mkdir()
    rows = []
    for seed_index, mean_return in returns.items():
        run_dir = group_dir / f"run-{seed_index}"
        run_dir.mkdir()
        with (run_dir / "final_summary.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "agent_seed": 100 + seed_index,
                    "env_seed": 200 + seed_index,
                    "validation": {
                        "mean_return_raw": mean_return,
                        "risk_adjusted_return_raw": mean_return - 1.0,
                        "std_return_raw": 2.0,
                        "mean_terminal_deviation": 0.0,
                        "mean_cumulative_cashflow": mean_return,
                        "mean_number_of_constrained_actions": 0.0,
                    },
                },
                file,
            )
        rows.append(
            {
                "seed_index": seed_index,
                "status": status,
                "run_dir": str(run_dir),
            }
        )

    runs_csv = group_dir / "runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["seed_index", "status", "run_dir"])
        writer.writeheader()
        writer.writerows(rows)
    return runs_csv
