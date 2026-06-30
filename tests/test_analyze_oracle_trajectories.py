"""Tests for perfect-foresight trajectory diagnostics."""

from __future__ import annotations

import json

import numpy as np

from gas_storage_rl.pretraining.analyze_oracle_trajectories import (
    action_counts_by_day,
    classify_action,
    default_output_dir,
    run_analysis,
    trajectory_rows_to_step_samples,
)


def test_classify_action_uses_tolerance_around_zero() -> None:
    """Small numerical residuals are classified as hold actions."""
    assert classify_action(1.0) == "+1"
    assert classify_action(-1.0) == "-1"
    assert classify_action(1e-8) == "0"
    assert classify_action(-1e-8) == "0"


def test_action_counts_by_absolute_day_include_ambiguity_metrics() -> None:
    """Per-day action counts include entropy, majority, and distinct actions."""
    samples = trajectory_rows_to_step_samples(
        [
            {
                "path_id": 0,
                "prices": [10.0, 20.0, 30.0],
                "actions": [1.0, 0.0, -1.0],
                "storage_levels": [0.0, 1.0, 1.0, 0.0],
            },
            {
                "path_id": 1,
                "prices": [11.0, 21.0, 31.0],
                "actions": [-1.0, 0.0, -1.0],
                "storage_levels": [2.0, 1.0, 1.0, 0.0],
            },
        ]
    )

    counts = action_counts_by_day(samples)
    day_0 = counts[counts["absolute_day"] == 0].iloc[0]
    day_1 = counts[counts["absolute_day"] == 1].iloc[0]
    day_2 = counts[counts["absolute_day"] == 2].iloc[0]

    assert day_0["+1"] == 1
    assert day_0["-1"] == 1
    assert day_0["0"] == 0
    assert day_0["n_distinct_actions"] == 2
    assert np.isclose(day_0["entropy"], 1.0)
    assert day_0["majority_share"] == 0.5
    assert day_1["majority_action"] == "0"
    assert day_1["majority_share"] == 1.0
    assert day_2["majority_action"] == "-1"


def test_step_samples_use_start_index_for_absolute_day() -> None:
    """Absolute day is the raw-path calendar index, not the episode step."""
    samples = trajectory_rows_to_step_samples(
        [
            {
                "path_id": 0,
                "start_index": 10,
                "prices": [10.0, 20.0],
                "actions": [1.0, -1.0],
                "storage_levels": [0.0, 1.0, 0.0],
            },
            {
                "path_id": 1,
                "start_date": "2001-01-21",
                "prices": [30.0, 40.0],
                "actions": [0.0, -1.0],
                "storage_levels": [1.0, 1.0, 0.0],
            },
        ]
    )

    assert samples["episode_day"].tolist() == [0, 1, 0, 1]
    assert samples["start_index"].tolist() == [10, 10, 20, 20]
    assert samples["absolute_day"].tolist() == [10, 11, 20, 21]


def test_default_output_dir_uses_benchmark_run_id() -> None:
    """Benchmark trajectory paths map to the matching diagnostics run folder."""
    output_dir = default_output_dir(
        "runs/benchmarks/20260625-debug-abcd/perfect_foresight_trajectories_pretrain.jsonl"
    )

    assert output_dir.as_posix() == "runs/diagnostics/20260625-debug-abcd"


def test_run_analysis_writes_csvs_and_selected_day_plots(tmp_path) -> None:
    """The CLI backend writes global and selected-day diagnostics."""
    run_dir = tmp_path / "runs" / "benchmarks" / "run-1"
    run_dir.mkdir(parents=True)
    trajectory_path = run_dir / "perfect_foresight_trajectories_pretrain.jsonl"
    rows = [
        {
            "split": "pretrain",
            "path_id": 0,
            "start_index": 0,
            "prices": [10.0, 20.0, 30.0],
            "actions": [1.0, 0.0, -1.0],
            "storage_levels": [0.0, 1.0, 1.0, 0.0],
            "target_inventory": 0.0,
        },
        {
            "split": "pretrain",
            "path_id": 1,
            "start_index": 0,
            "prices": [12.0, 19.0, 33.0],
            "actions": [-1.0, 0.0, -1.0],
            "storage_levels": [2.0, 1.0, 1.0, 0.0],
            "target_inventory": 0.0,
        },
    ]
    with trajectory_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")

    output_dir = run_analysis(trajectory_path, selected_days=[0])

    assert output_dir == tmp_path / "runs" / "diagnostics" / "run-1"
    assert (output_dir / "step_samples.csv").exists()
    assert (output_dir / "action_counts_by_day.csv").exists()
    assert (output_dir / "ambiguous_days.csv").exists()
    assert (output_dir / "feature_stats_by_day_action.csv").exists()
    assert (output_dir / "action_distribution_by_day.png").exists()
    assert (output_dir / "feature_stats_selected_days.csv").exists()
    assert (output_dir / "day_000_action_counts.png").exists()
    assert (output_dir / "day_000_price_by_action.png").exists()
    assert (output_dir / "day_000_storage_by_action.png").exists()
    assert (output_dir / "day_000_price_vs_storage.png").exists()
