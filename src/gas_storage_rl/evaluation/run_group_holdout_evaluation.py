"""Manual holdout evaluation for complete experiment groups."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.backtest import build_backtest_evaluation_dataset
from gas_storage_rl.evaluation.metrics import interquartile_mean
from gas_storage_rl.evaluation.run_holdout_evaluation import (
    MODEL_CHECKPOINTS,
    evaluate_holdout_run,
    perfect_foresight_reference_rows,
)
from gas_storage_rl.logging.progress import CliProgress
from gas_storage_rl.plotting.statistics import bootstrap_percentile_statistic_ci
from gas_storage_rl.training.config import build_environment, load_config

AGGREGATED_RUN_METRICS = (
    "mean_return_scaled",
    "mean_return_raw",
    "median_return_raw",
    "std_return_raw",
    "min_return_raw",
    "max_return_raw",
    "interquartile_mean_return_raw",
    "mean_terminal_deviation",
    "mean_cumulative_cashflow",
    "mean_terminal_penalty",
    "mean_number_of_constrained_actions",
    "mean_number_of_physical_clipped_actions",
    "mean_number_of_terminal_feasibility_clipped_actions",
    "mean_number_of_physical_only_clipped_actions",
    "mean_number_of_terminal_only_clipped_actions",
    "mean_number_of_physical_and_terminal_clipped_actions",
    "mean_perfect_foresight_return_raw",
    "mean_perfect_foresight_ratio",
    "mean_optimality_gap",
)


def run_group_holdout_evaluation(
    group_dir: str | Path,
    split: str = "test",
    *,
    model_checkpoint: str = "best_validation",
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluates all completed runs in an experiment group on a holdout split.

    Args:
        group_dir: Experiment-group directory containing ``runs.csv``.
        split: Dataset split to evaluate.
        model_checkpoint: Named model checkpoint to load from every run.
        n_bootstrap: Number of percentile bootstrap samples for IQM intervals.
        bootstrap_seed: Seed used for deterministic bootstrap intervals.

    Returns:
        Group holdout summary.
    """
    if model_checkpoint not in MODEL_CHECKPOINTS:
        raise ValueError(f"Unsupported model checkpoint: {model_checkpoint}")
    group_path = Path(group_dir)
    runs = _read_csv(group_path / "runs.csv")
    group_config = load_config(group_path / "group_config.json")
    progress = CliProgress("run_group_holdout_evaluation", total=3 + len(runs))

    progress.step(1, "building group reference dataset")
    references = _build_group_references(group_config, split)
    reference_file = group_path / f"perfect_foresight_references_{split}.jsonl"
    _write_jsonl(reference_file, references)

    progress.step(2, "evaluating experiment runs")
    run_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(runs, start=1):
        progress.step(2 + index, f"evaluating seed={source_row.get('seed_index')}")
        run_rows.append(
            _evaluate_group_run(
                source_row,
                split,
                model_checkpoint,
                references,
                episode_rows,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
            )
        )
        _write_csv(group_path / f"holdout_{split}_runs.csv", run_rows)
        _write_csv(group_path / f"holdout_{split}_episode_metrics.csv", episode_rows)

    progress.step(3 + len(runs), "aggregating seed metrics")
    summary = _group_summary(
        group_path,
        group_config,
        split,
        model_checkpoint,
        reference_file,
        references,
        run_rows,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    _write_json(group_path / f"holdout_{split}_summary.json", summary)
    progress.finish("done")
    return summary


def _build_group_references(
    config: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    """Builds path-wise perfect-foresight references once for a group."""
    dataset, storage_params, env_kwargs = build_environment(config)
    if split == "backtest":
        dataset = build_backtest_evaluation_dataset(config, dataset)
    env = GasStorageEnv(dataset, split, storage_params, **env_kwargs)
    return perfect_foresight_reference_rows(
        dataset,
        split,
        storage_params,
        env.lambda_terminal,
    )


def _evaluate_group_run(
    source_row: dict[str, str],
    split: str,
    model_checkpoint: str,
    references: list[dict[str, Any]],
    group_episode_rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Evaluates one group member and appends its episode rows."""
    output_row: dict[str, Any] = {
        "experiment_group_id": source_row.get("experiment_group_id", ""),
        "algorithm_name": source_row.get("algorithm_name", ""),
        "seed_index": source_row.get("seed_index", ""),
        "master_seed": source_row.get("master_seed", ""),
        "dataset_seed": source_row.get("dataset_seed", ""),
        "eval_seed": source_row.get("eval_seed", ""),
        "env_seed": source_row.get("env_seed", ""),
        "agent_seed": source_row.get("agent_seed", ""),
        "source_run_status": source_row.get("status", ""),
        "holdout_status": "running",
        "run_dir": source_row.get("run_dir", ""),
        "model_checkpoint": model_checkpoint,
        "error": "",
    }
    run_dir = source_row.get("run_dir", "")
    if not run_dir or source_row.get("status") == "failed":
        output_row["holdout_status"] = "skipped"
        output_row["error"] = "No completed experiment run to evaluate"
        return output_row

    try:
        metrics = evaluate_holdout_run(
            run_dir,
            split,
            model_checkpoint=model_checkpoint,
            write_final_episode_metrics=True,
            perfect_foresight_references=references,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
        output_row.update(metrics)
        output_row["holdout_status"] = "completed"
        _append_group_episode_rows(
            group_episode_rows,
            Path(run_dir) / f"final_episode_metrics_{split}.csv",
            output_row,
        )
    except Exception as error:  # Continue so one bad run does not block the group.
        output_row["holdout_status"] = "failed"
        output_row["error"] = f"{type(error).__name__}: {error}"
    return output_row


def _append_group_episode_rows(
    output: list[dict[str, Any]],
    episode_file: Path,
    run_row: dict[str, Any],
) -> None:
    """Appends one run's episode metrics with group identifiers."""
    for row in _read_csv(episode_file):
        output.append(
            {
                "experiment_group_id": run_row["experiment_group_id"],
                "seed_index": run_row["seed_index"],
                "run_dir": run_row["run_dir"],
                "model_checkpoint": run_row["model_checkpoint"],
                **row,
            }
        )


def _group_summary(
    group_dir: Path,
    config: dict[str, Any],
    split: str,
    model_checkpoint: str,
    reference_file: Path,
    references: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Builds group-level metadata and seed-axis metric aggregates."""
    completed_rows = [
        row for row in run_rows if row.get("holdout_status") == "completed"
    ]
    aggregates = {
        metric: _aggregate_metric(
            [row.get(metric) for row in completed_rows],
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
        for metric in AGGREGATED_RUN_METRICS
    }
    summary = {
        "experiment_group_id": group_dir.name,
        "split": split,
        "model_checkpoint": model_checkpoint,
        "n_group_runs": len(run_rows),
        "n_evaluated_runs": len(completed_rows),
        "n_failed_evaluations": sum(
            row.get("holdout_status") == "failed" for row in run_rows
        ),
        "n_skipped_evaluations": sum(
            row.get("holdout_status") == "skipped" for row in run_rows
        ),
        "n_perfect_foresight_references": len(references),
        "perfect_foresight_reference_file": reference_file.name,
        "dataset_seed": int(config["seeds"]["dataset_seed"]),
        "eval_seed": int(config["seeds"]["eval_seed"]),
        "iqm_bootstrap_n": int(n_bootstrap),
        "iqm_bootstrap_confidence_level": 0.95,
        "bootstrap_seed": int(bootstrap_seed),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metric_aggregates_over_seed": aggregates,
    }
    summary.update(_flat_primary_return_aggregate(aggregates["mean_return_raw"]))
    return summary


def _aggregate_metric(
    values: list[Any],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, float | int | None]:
    """Computes descriptive statistics for one run-level metric over seeds."""
    sample = np.asarray([_to_float(value) for value in values], dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "interquartile_mean": None,
            "interquartile_mean_ci_lower": None,
            "interquartile_mean_ci_upper": None,
        }
    ci_lower, ci_upper = bootstrap_percentile_statistic_ci(
        sample,
        statistic=interquartile_mean,
        n_bootstrap=n_bootstrap,
        random_seed=bootstrap_seed,
    )
    return {
        "n": int(sample.size),
        "mean": float(np.mean(sample)),
        "median": float(np.median(sample)),
        "std": float(np.std(sample)),
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
        "interquartile_mean": interquartile_mean(sample),
        "interquartile_mean_ci_lower": ci_lower,
        "interquartile_mean_ci_upper": ci_upper,
    }


def _flat_primary_return_aggregate(
    aggregate: dict[str, float | int | None],
) -> dict[str, float | int | None]:
    """Returns backward-readable top-level return aggregate fields."""
    return {
        "mean_return_raw_over_seed": aggregate["mean"],
        "median_return_raw_over_seed": aggregate["median"],
        "std_return_raw_over_seed": aggregate["std"],
        "min_return_raw_over_seed": aggregate["min"],
        "max_return_raw_over_seed": aggregate["max"],
        "interquartile_mean_return_raw_over_seed": aggregate["interquartile_mean"],
        "interquartile_mean_return_raw_over_seed_ci_lower": aggregate[
            "interquartile_mean_ci_lower"
        ],
        "interquartile_mean_return_raw_over_seed_ci_upper": aggregate[
            "interquartile_mean_ci_upper"
        ],
    }


def _to_float(value: Any) -> float:
    """Converts CSV and JSON scalar values to float when possible."""
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Reads a CSV file as dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes one JSON document."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes one JSON object per line."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, default=str))
            file.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes rows with a union of all encountered fields."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    """Runs holdout evaluation for an experiment group."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-dir", required=True)
    parser.add_argument(
        "--split",
        default="test",
        choices=["validation", "test", "backtest"],
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHECKPOINTS),
        default="best_validation",
        help="Model checkpoint to evaluate. Defaults to best_validation_model.zip.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    summary = run_group_holdout_evaluation(
        args.group_dir,
        args.split,
        model_checkpoint=args.model,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary, indent=2, default=str))
    if summary["n_failed_evaluations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
