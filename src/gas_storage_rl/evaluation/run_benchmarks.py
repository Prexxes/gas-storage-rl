"""Command-line runner for benchmark policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from gas_storage_rl.baselines.lsmc import LSMCBenchmark
from gas_storage_rl.baselines.oracle_cloned_policy import OracleClonedPolicy
from gas_storage_rl.baselines.perfect_foresight import PerfectForesightBaseline
from gas_storage_rl.baselines.random_policy import RandomPolicy
from gas_storage_rl.baselines.rule_based_policy import RuleBasedPolicy
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.logging.progress import CliProgress
from gas_storage_rl.pretraining.behavior_cloning import trajectory_rows_to_samples
from gas_storage_rl.training.config import build_environment, load_config

DEFAULT_SPLITS = ("train", "validation")
ORACLE_CLONING_TRAINING_SPLITS = ("pretrain", "train")
ORACLE_CLONING_EVALUATION_SPLITS = {"validation", "test"}


def config_hash(config: dict[str, Any]) -> str:
    """Returns a short stable hash for a loaded configuration."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]


def create_benchmark_run_dir(
    base_dir: str | Path,
    config_name: str,
    hash_value: str,
) -> tuple[Path, str]:
    """Creates a benchmark run directory with a human-readable run id."""
    safe_config_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in config_name
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_path = Path(base_dir) / "benchmarks"
    run_id = f"{timestamp}-{safe_config_name}-{hash_value}"
    run_dir = base_path / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{timestamp}-{safe_config_name}-{hash_value}-{suffix}"
        run_dir = base_path / run_id
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes a JSON payload to disk."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes benchmark metrics in long CSV format."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes one JSON object per line."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, default=str))
            file.write("\n")


def benchmark_metadata(
    run_id: str,
    config_name: str,
    hash_value: str,
    splits: list[str],
    writes_perfect_foresight_trajectories: bool,
    includes_oracle_cloned_policy: bool,
) -> dict[str, Any]:
    """Returns reproducibility metadata for a benchmark run."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_commit = None
    return {
        "run_id": run_id,
        "config_name": config_name,
        "config_hash": hash_value,
        "splits": splits,
        "contains_test_split": "test" in splits,
        "writes_perfect_foresight_trajectories": (
            writes_perfect_foresight_trajectories
        ),
        "includes_oracle_cloned_policy": includes_oracle_cloned_policy,
        "git_commit_hash": git_commit,
        "python_version": sys.version,
        "platform": platform.platform(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def validate_splits(dataset: Any, splits: list[str]) -> None:
    """Raises a clear error when requested splits are unavailable."""
    available = set(dataset.paths_by_split)
    missing = [split for split in splits if split not in available]
    if missing:
        raise ValueError(
            "Unknown benchmark split(s) "
            f"{missing}. Available splits are {sorted(available)}."
        )
    if "train" not in available:
        raise ValueError("Benchmark policies require a train split for calibration.")


def enrich_metrics(
    metrics: dict[str, Any],
    benchmark: str,
    split: str,
    run_id: str,
    config_name: str,
    hash_value: str,
) -> dict[str, Any]:
    """Adds benchmark-run identifiers to a metrics dictionary."""
    enriched = {
        "run_id": run_id,
        "config_name": config_name,
        "config_hash": hash_value,
        "split": split,
        "benchmark": benchmark,
    }
    enriched.update(metrics)
    return enriched


def _date_ranges_for_split(dataset: Any, split: str) -> list[dict[str, str]] | None:
    """Returns serialized date ranges for a split when available."""
    if dataset.date_ranges_by_split is None:
        return None
    return dataset.date_ranges_by_split.get(split)


def fit_oracle_cloned_policy(
    dataset: Any,
    storage_params: Any,
    env_kwargs: dict[str, Any],
    perfect_foresight: PerfectForesightBaseline,
    observation_dim: int,
    config: dict[str, Any],
) -> tuple[OracleClonedPolicy, dict[str, Any]]:
    """Fits a policy on pretrain and train perfect-foresight trajectories."""
    available = set(dataset.paths_by_split)
    missing = [
        split for split in ORACLE_CLONING_TRAINING_SPLITS if split not in available
    ]
    if missing:
        raise ValueError(
            "Oracle-cloned policy requires perfect-foresight training splits "
            f"{list(ORACLE_CLONING_TRAINING_SPLITS)}. Missing: {missing}."
        )

    trajectories = []
    trajectory_counts = {}
    for split in ORACLE_CLONING_TRAINING_SPLITS:
        split_paths = dataset.get_paths(split)
        split_inventories = dataset.get_initial_inventories(
            split,
            storage_params.initial_inventory,
        )
        split_trajectories = perfect_foresight.solve_paths(
            split_paths,
            split,
            _date_ranges_for_split(dataset, split),
            split_inventories,
            split_inventories,
        )
        trajectories.extend(split_trajectories)
        trajectory_counts[split] = len(split_trajectories)

    observations, actions = trajectory_rows_to_samples(
        trajectories,
        capacity=storage_params.capacity,
        price_scale=float(env_kwargs.get("price_scale", 50.0)),
    )
    evaluation_config = config.get("evaluation_config", {})
    seed = int(config["seeds"].get("agent_seed", config["seeds"]["eval_seed"]))
    policy = OracleClonedPolicy(
        observation_dim=observation_dim,
        hidden_sizes=tuple(
            evaluation_config.get("oracle_cloned_policy_hidden_sizes", [64, 64])
        ),
        seed=seed,
        device=str(evaluation_config.get("oracle_cloned_policy_device", "cpu")),
    )
    history = policy.fit(
        observations,
        actions,
        epochs=int(evaluation_config.get("oracle_cloned_policy_epochs", 20)),
        batch_size=int(evaluation_config.get("oracle_cloned_policy_batch_size", 256)),
        learning_rate=float(
            evaluation_config.get("oracle_cloned_policy_learning_rate", 1e-3)
        ),
        seed=seed,
    )
    return policy, {
        "training_splits": list(ORACLE_CLONING_TRAINING_SPLITS),
        "trajectory_counts": trajectory_counts,
        "n_samples": int(len(observations)),
        "final_imitation_loss": float(history[-1]["loss"]),
        "epochs": int(history[-1]["epoch"]),
    }


def run_benchmarks(
    config: dict[str, Any],
    config_name: str,
    splits: list[str],
    write_perfect_foresight_trajectories: bool = False,
    include_oracle_cloned_policy: bool = False,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Runs all benchmark policies and returns metrics by split."""
    progress_total = 3 + len(splits) + int(include_oracle_cloned_policy)
    progress = CliProgress("run_benchmarks", total=progress_total, enabled=show_progress)
    progress_step = 1
    progress.step(progress_step, "building environment")
    dataset, storage_params, env_kwargs = build_environment(config)
    validate_splits(dataset, splits)

    progress_step += 1
    progress.step(progress_step, "fitting train-calibrated baselines")
    train_paths = dataset.get_paths("train")
    train_env = GasStorageEnv(dataset, "train", storage_params, **env_kwargs)
    rule_policy = RuleBasedPolicy.from_training_prices(
        train_paths,
        capacity=storage_params.capacity,
        episode_length=dataset.episode_length,
        injection_rate=storage_params.injection_rate,
        withdrawal_rate=storage_params.withdrawal_rate,
    )
    action_grid = np.asarray(
        config.get("evaluation_config", {}).get("lsmc_action_grid", [-1.0, 0.0, 1.0]),
        dtype=np.float64,
    )
    lsmc = LSMCBenchmark(
        storage_params,
        train_env.lambda_terminal,
        action_grid=action_grid,
    ).fit(
        train_paths,
        dataset.get_start_dates("train"),
        dataset.get_initial_inventories(
            "train",
            storage_params.initial_inventory,
        ),
    )
    perfect_foresight = PerfectForesightBaseline(
        storage_params,
        train_env.lambda_terminal,
    )
    oracle_cloned_policy = None
    oracle_training_summary = None
    if include_oracle_cloned_policy:
        progress_step += 1
        progress.step(progress_step, "training oracle-cloned policy")
        oracle_cloned_policy, oracle_training_summary = fit_oracle_cloned_policy(
            dataset,
            storage_params,
            env_kwargs,
            perfect_foresight,
            int(train_env.observation_space.shape[0]),
            config,
        )

    output = {
        "config_name": config_name,
        "splits": {},
    }
    if oracle_training_summary is not None:
        output["oracle_cloned_policy_training"] = oracle_training_summary
    if write_perfect_foresight_trajectories:
        output["_perfect_foresight_trajectories"] = {}
    for split in splits:
        progress_step += 1
        progress.step(progress_step, f"evaluating split={split}")
        env = GasStorageEnv(dataset, split, storage_params, **env_kwargs)
        random_policy = RandomPolicy(seed=config["seeds"]["eval_seed"])
        random_metrics, _ = evaluate_policy_on_paths(env, random_policy)
        rule_metrics, _ = evaluate_policy_on_paths(env, rule_policy)
        split_paths = dataset.get_paths(split)
        split_inventories = dataset.get_initial_inventories(
            split,
            storage_params.initial_inventory,
        )
        output["splits"][split] = {
            "random": random_metrics,
            "rule_based": rule_metrics,
            "lsmc": lsmc.evaluate(
                split_paths,
                dataset.get_start_dates(split),
                split_inventories,
            ),
            "perfect_foresight": perfect_foresight.evaluate_paths(
                split_paths,
                split_inventories,
                split_inventories,
            ),
        }
        if (
            oracle_cloned_policy is not None
            and split in ORACLE_CLONING_EVALUATION_SPLITS
        ):
            oracle_metrics, _ = evaluate_policy_on_paths(env, oracle_cloned_policy)
            oracle_metrics["imitation_training_samples"] = float(
                oracle_training_summary["n_samples"]
            )
            oracle_metrics["final_imitation_loss"] = float(
                oracle_training_summary["final_imitation_loss"]
            )
            output["splits"][split]["oracle_cloned_policy"] = oracle_metrics
        if write_perfect_foresight_trajectories:
            output["_perfect_foresight_trajectories"][split] = (
                perfect_foresight.solve_paths(
                    split_paths,
                    split,
                    _date_ranges_for_split(dataset, split),
                    split_inventories,
                    split_inventories,
                )
            )
    progress_step += 1
    progress.step(progress_step, "assembled metrics")
    progress.finish("done")
    return output


def log_benchmark_results(
    output: dict[str, Any],
    config: dict[str, Any],
    config_name: str,
    splits: list[str],
    write_perfect_foresight_trajectories: bool = False,
    include_oracle_cloned_policy: bool = False,
) -> Path:
    """Persists benchmark metrics as config, metadata, JSON, and CSV files."""
    hash_value = config_hash(config)
    run_dir, run_id = create_benchmark_run_dir(
        config["logging_config"]["run_dir"],
        config_name,
        hash_value,
    )
    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "metadata.json",
        benchmark_metadata(
            run_id,
            config_name,
            hash_value,
            splits,
            write_perfect_foresight_trajectories,
            include_oracle_cloned_policy,
        ),
    )

    csv_rows = []
    trajectory_counts = {}
    trajectory_file_names = {}
    for split, benchmarks in output["splits"].items():
        split_payload = {"split": split, "benchmarks": benchmarks}
        write_json(run_dir / f"benchmark_metrics_{split}.json", split_payload)
        for benchmark, metrics in benchmarks.items():
            csv_rows.append(
                enrich_metrics(
                    metrics,
                    benchmark,
                    split,
                    run_id,
                    config_name,
                    hash_value,
                )
            )
        if write_perfect_foresight_trajectories:
            trajectories = output["_perfect_foresight_trajectories"][split]
            file_name = f"perfect_foresight_trajectories_{split}.jsonl"
            write_jsonl(run_dir / file_name, trajectories)
            trajectory_counts[split] = len(trajectories)
            trajectory_file_names[split] = file_name
    write_metrics_csv(run_dir / "benchmark_metrics.csv", csv_rows)
    output["run_dir"] = str(run_dir)
    output["run_id"] = run_id
    if write_perfect_foresight_trajectories:
        output.pop("_perfect_foresight_trajectories", None)
        output["perfect_foresight_trajectory_counts"] = trajectory_counts
        output["perfect_foresight_trajectory_files"] = trajectory_file_names
    return run_dir


def main() -> None:
    """Runs random, rule-based, LSMC, and perfect-foresight benchmarks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=["pretrain", "train", "validation", "test", "backtest"],
        help=(
            "Dataset split to benchmark. Can be passed multiple times. "
            "Defaults to train and validation; test only runs when explicit."
        ),
    )
    parser.add_argument(
        "--write-perfect-foresight-trajectories",
        action="store_true",
        help=(
            "Write per-path perfect-foresight prices, actions, storage levels, "
            "objective values, terminal deviations, success flags, and dates."
        ),
    )
    parser.add_argument(
        "--include-oracle-cloned-policy",
        action="store_true",
        help=(
            "Train an observation-only neural policy on pretrain and train "
            "perfect-foresight trajectories, then report it on validation/test."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    splits = args.split or list(DEFAULT_SPLITS)
    config_name = Path(args.config).stem
    output = run_benchmarks(
        config,
        config_name,
        splits,
        write_perfect_foresight_trajectories=args.write_perfect_foresight_trajectories,
        include_oracle_cloned_policy=args.include_oracle_cloned_policy,
        show_progress=True,
    )
    log_benchmark_results(
        output,
        config,
        config_name,
        splits,
        write_perfect_foresight_trajectories=args.write_perfect_foresight_trajectories,
        include_oracle_cloned_policy=args.include_oracle_cloned_policy,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
