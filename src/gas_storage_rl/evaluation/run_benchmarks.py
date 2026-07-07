"""Command-line runner for benchmark policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
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
from gas_storage_rl.evaluation.metrics import (
    summarize_episode_infos,
    summarize_evaluation,
)
from gas_storage_rl.logging.progress import CliProgress
from gas_storage_rl.pretraining.behavior_cloning import trajectory_rows_to_samples
from gas_storage_rl.training.config import build_environment, load_config

DEFAULT_SPLITS = ("train", "validation")
ORACLE_CLONING_TRAINING_SPLITS = ("pretrain", "train")
ORACLE_CLONING_EVALUATION_SPLITS = {"validation", "test"}
METHOD_TYPES = {
    "random": "stochastic_baseline",
    "rule_based": "deterministic_baseline",
    "lsmc": "fit_baseline",
    "perfect_foresight": "oracle_upper_bound",
    "oracle_cloned_policy": "imitation_baseline",
}


class LSMCPolicyAdapter:
    """Adapts an LSMC benchmark to the SB3-style policy interface."""

    def __init__(self, benchmark: LSMCBenchmark, env: GasStorageEnv) -> None:
        """Initializes the adapter."""
        self.benchmark = benchmark
        self.env = env

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Returns the LSMC action for the environment's current state."""
        del observation, deterministic
        start_date = self.env.dataset.get_start_dates(self.env.split)[
            self.env.current_path_id
        ]
        action = self.benchmark.predict_action(
            storage_level=self.env.storage_level,
            price=float(self.env.current_path[self.env.current_step]),
            current_step=self.env.current_step,
            horizon=self.env.episode_length,
            start_date=start_date,
            target_inventory=self.env.target_terminal_inventory,
        )
        return np.array([action], dtype=np.float32), None


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


def timeline_steps(total_timesteps: int | None, eval_freq: int | None) -> list[int]:
    """Returns evaluation steps used for benchmark reference lines."""
    if total_timesteps is None:
        return []
    if total_timesteps < 0:
        raise ValueError("timeline_total_timesteps must be non-negative")
    if eval_freq is None or eval_freq <= 0:
        return [int(total_timesteps)]
    steps = list(range(0, int(total_timesteps) + 1, int(eval_freq)))
    if steps[-1] != int(total_timesteps):
        steps.append(int(total_timesteps))
    return steps


def benchmark_timeline_rows(
    output: dict[str, Any],
    steps: list[int],
    config_name: str,
    hash_value: str,
) -> list[dict[str, Any]]:
    """Expands benchmark metrics over fixed training-step coordinates."""
    rows = []
    for split, benchmarks in output["splits"].items():
        for benchmark, metrics in benchmarks.items():
            for step in steps:
                row = {
                    "config_name": config_name,
                    "config_hash": hash_value,
                    "split": split,
                    "method": benchmark,
                    "benchmark": benchmark,
                    "method_type": METHOD_TYPES.get(benchmark, "benchmark"),
                    "total_training_env_steps": int(step),
                    "is_baseline_reference": True,
                }
                row.update(metrics)
                row["total_training_env_steps"] = int(step)
                rows.append(row)
    return rows


def trajectory_episode_rows(
    trajectories: list[dict[str, Any]],
    method: str,
    split: str,
    date_ranges: list[dict[str, str]] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Converts environment trajectories to one row per evaluated episode."""
    rows = []
    for trajectory in trajectories:
        infos = trajectory["infos"]
        if not infos:
            continue
        summary = summarize_episode_infos(infos)
        first_info = infos[0]
        last_info = infos[-1]
        rows.append(
            {
                "method": method,
                "benchmark": method,
                "method_type": METHOD_TYPES.get(method, "benchmark"),
                "split": split,
                "path_id": int(trajectory["path_id"]),
                "start_index": int(first_info["start_index"]),
                "start_date": _start_date_for_path(
                    date_ranges,
                    int(trajectory["path_id"]),
                ),
                "initial_inventory": float(first_info["initial_inventory"]),
                "target_inventory": float(last_info["target_terminal_inventory"]),
                "seed": seed,
                **summary,
            }
        )
    return rows


def perfect_foresight_episode_rows(
    trajectories: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    """Converts perfect-foresight solution trajectories to episode metric rows."""
    rows = []
    for trajectory in trajectories:
        rows.append(
            {
                "method": "perfect_foresight",
                "benchmark": "perfect_foresight",
                "method_type": METHOD_TYPES["perfect_foresight"],
                "split": split,
                "path_id": int(trajectory["path_id"]),
                "start_index": None,
                "start_date": trajectory.get("start_date"),
                "initial_inventory": float(trajectory["initial_inventory"]),
                "target_inventory": float(trajectory["target_inventory"]),
                "seed": None,
                "episode_return_raw": float(trajectory["objective_value"]),
                "episode_return_scaled": None,
                "episode_length": float(len(trajectory["actions"])),
                "final_storage_level": float(trajectory["storage_levels"][-1]),
                "terminal_deviation": float(trajectory["terminal_deviation"]),
                "terminal_penalty": None,
                "cumulative_cashflow": None,
                "number_of_constrained_actions": None,
                "total_injected_volume": float(
                    np.sum(np.maximum(trajectory["actions"], 0.0))
                ),
                "total_withdrawn_volume": float(
                    -np.sum(np.minimum(trajectory["actions"], 0.0))
                ),
                "mean_storage_level": float(np.mean(trajectory["storage_levels"][1:])),
                "mean_action": float(np.mean(trajectory["actions"])),
                "success": bool(trajectory["success"]),
            }
        )
    return rows


def _start_date_for_path(
    date_ranges: list[dict[str, str]] | None,
    path_id: int,
) -> str | None:
    """Returns the serialized start date for a path id if available."""
    if date_ranges is None:
        return None
    return date_ranges[path_id].get("start_date")


def benchmark_metadata(
    run_id: str,
    config_name: str,
    hash_value: str,
    splits: list[str],
    writes_perfect_foresight_trajectories: bool,
    includes_oracle_cloned_policy: bool,
    writes_final_episode_metrics: bool,
    timeline_total_timesteps: int | None,
    timeline_eval_freq: int | None,
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
        "writes_final_episode_metrics": writes_final_episode_metrics,
        "timeline_total_timesteps": timeline_total_timesteps,
        "timeline_eval_freq": timeline_eval_freq,
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
    write_final_episode_metrics: bool = False,
    random_policy_seeds: list[int] | None = None,
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
        config.get("evaluation_config", {}).get(
            "lsmc_action_grid",
            [-1.0, -0.5, 0.0, 0.5, 1.0],
        ),
        dtype=np.float64,
    )
    lsmc = LSMCBenchmark(
        storage_params,
        train_env.lambda_terminal,
        action_grid=action_grid,
        n_inventory_levels=int(
            config.get("evaluation_config", {}).get("lsmc_n_inventory_levels", 21)
        ),
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
    random_policy_seeds = random_policy_seeds or [int(config["seeds"]["eval_seed"])]
    if oracle_training_summary is not None:
        output["oracle_cloned_policy_training"] = oracle_training_summary
    output["_benchmark_artifacts"] = {"lsmc": lsmc}
    if oracle_cloned_policy is not None:
        output["_benchmark_artifacts"]["oracle_cloned_policy"] = oracle_cloned_policy
    if write_perfect_foresight_trajectories:
        output["_perfect_foresight_trajectories"] = {}
    if write_final_episode_metrics:
        output["_final_episode_metrics"] = {}
    for split in splits:
        progress_step += 1
        progress.step(progress_step, f"evaluating split={split}")
        env = GasStorageEnv(dataset, split, storage_params, **env_kwargs)
        date_ranges = _date_ranges_for_split(dataset, split)
        final_episode_rows = []

        random_episode_summaries = []
        for random_seed in random_policy_seeds:
            random_policy = RandomPolicy(seed=random_seed)
            _, random_trajectories = evaluate_policy_on_paths(env, random_policy)
            random_episode_summaries.extend(
                [
                    summarize_episode_infos(trajectory["infos"])
                    for trajectory in random_trajectories
                ]
            )
            if write_final_episode_metrics:
                final_episode_rows.extend(
                    trajectory_episode_rows(
                        random_trajectories,
                        "random",
                        split,
                        date_ranges,
                        random_seed,
                    )
                )
        random_metrics = summarize_evaluation(
            random_episode_summaries,
            split,
        )
        if len(random_policy_seeds) > 1:
            random_metrics["n_action_seeds"] = float(len(random_policy_seeds))

        rule_metrics, rule_trajectories = evaluate_policy_on_paths(env, rule_policy)
        if write_final_episode_metrics:
            final_episode_rows.extend(
                trajectory_episode_rows(
                    rule_trajectories,
                    "rule_based",
                    split,
                    date_ranges,
                )
            )
        lsmc_metrics, lsmc_trajectories = evaluate_policy_on_paths(
            env,
            LSMCPolicyAdapter(lsmc, env),
        )
        if write_final_episode_metrics:
            final_episode_rows.extend(
                trajectory_episode_rows(
                    lsmc_trajectories,
                    "lsmc",
                    split,
                    date_ranges,
                )
            )
        split_paths = dataset.get_paths(split)
        split_inventories = dataset.get_initial_inventories(
            split,
            storage_params.initial_inventory,
        )
        perfect_foresight_trajectories = perfect_foresight.solve_paths(
            split_paths,
            split,
            date_ranges,
            split_inventories,
            split_inventories,
        )
        if write_final_episode_metrics:
            final_episode_rows.extend(
                perfect_foresight_episode_rows(
                    perfect_foresight_trajectories,
                    split,
                )
            )
        output["splits"][split] = {
            "random": random_metrics,
            "rule_based": rule_metrics,
            "lsmc": lsmc_metrics,
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
            oracle_metrics, oracle_trajectories = evaluate_policy_on_paths(
                env,
                oracle_cloned_policy,
            )
            oracle_metrics["imitation_training_samples"] = float(
                oracle_training_summary["n_samples"]
            )
            oracle_metrics["final_imitation_loss"] = float(
                oracle_training_summary["final_imitation_loss"]
            )
            output["splits"][split]["oracle_cloned_policy"] = oracle_metrics
            if write_final_episode_metrics:
                final_episode_rows.extend(
                    trajectory_episode_rows(
                        oracle_trajectories,
                        "oracle_cloned_policy",
                        split,
                        date_ranges,
                    )
                )
        if write_perfect_foresight_trajectories:
            output["_perfect_foresight_trajectories"][split] = (
                perfect_foresight_trajectories
            )
        if write_final_episode_metrics:
            output["_final_episode_metrics"][split] = final_episode_rows
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
    write_final_episode_metrics: bool = False,
    timeline_total_timesteps: int | None = None,
    timeline_eval_freq: int | None = None,
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
            write_final_episode_metrics,
            timeline_total_timesteps,
            timeline_eval_freq,
        ),
    )

    csv_rows = []
    final_episode_file_names = {}
    trajectory_counts = {}
    trajectory_file_names = {}
    artifact_file_names = {}
    artifacts = output.get("_benchmark_artifacts", {})
    if "lsmc" in artifacts:
        file_name = "lsmc_policy.pkl"
        with (run_dir / file_name).open("wb") as file:
            pickle.dump(artifacts["lsmc"], file)
        artifact_file_names["lsmc"] = file_name
    if "oracle_cloned_policy" in artifacts:
        file_name = "oracle_cloned_policy.pt"
        artifacts["oracle_cloned_policy"].save(
            run_dir / file_name,
            metadata=output.get("oracle_cloned_policy_training", {}),
        )
        artifact_file_names["oracle_cloned_policy"] = file_name
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
        if write_final_episode_metrics:
            episode_rows = output["_final_episode_metrics"][split]
            file_name = f"final_episode_metrics_{split}.csv"
            write_metrics_csv(run_dir / file_name, episode_rows)
            final_episode_file_names[split] = file_name
    write_metrics_csv(run_dir / "benchmark_metrics.csv", csv_rows)
    steps = timeline_steps(timeline_total_timesteps, timeline_eval_freq)
    if steps:
        write_metrics_csv(
            run_dir / "benchmark_evaluations.csv",
            benchmark_timeline_rows(output, steps, config_name, hash_value),
        )
    output["run_dir"] = str(run_dir)
    output["run_id"] = run_id
    if write_perfect_foresight_trajectories:
        output.pop("_perfect_foresight_trajectories", None)
        output["perfect_foresight_trajectory_counts"] = trajectory_counts
        output["perfect_foresight_trajectory_files"] = trajectory_file_names
    if write_final_episode_metrics:
        output.pop("_final_episode_metrics", None)
        output["final_episode_metric_files"] = final_episode_file_names
    output.pop("_benchmark_artifacts", None)
    output["benchmark_artifact_files"] = artifact_file_names
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
    parser.add_argument(
        "--write-final-episode-metrics",
        action="store_true",
        help=(
            "Write one final episode-metrics CSV per requested split for direct "
            "per-path benchmark comparisons."
        ),
    )
    parser.add_argument(
        "--timeline-total-timesteps",
        type=int,
        help=(
            "Total RL training steps used to expand benchmark metrics into "
            "benchmark_evaluations.csv. Defaults to training_config.total_timesteps."
        ),
    )
    parser.add_argument(
        "--timeline-eval-freq",
        type=int,
        help=(
            "Step interval used for benchmark_evaluations.csv. Defaults to "
            "training_config.eval_freq."
        ),
    )
    parser.add_argument(
        "--random-policy-seed",
        action="append",
        type=int,
        help=(
            "Random-policy action seed. Can be passed multiple times. Defaults "
            "to seeds.eval_seed."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    splits = args.split or list(DEFAULT_SPLITS)
    config_name = Path(args.config).stem
    training_config = config.get("training_config", {})
    timeline_total_timesteps = (
        args.timeline_total_timesteps
        if args.timeline_total_timesteps is not None
        else training_config.get("total_timesteps")
    )
    timeline_eval_freq = (
        args.timeline_eval_freq
        if args.timeline_eval_freq is not None
        else training_config.get("eval_freq")
    )
    output = run_benchmarks(
        config,
        config_name,
        splits,
        write_perfect_foresight_trajectories=args.write_perfect_foresight_trajectories,
        include_oracle_cloned_policy=args.include_oracle_cloned_policy,
        write_final_episode_metrics=args.write_final_episode_metrics,
        random_policy_seeds=args.random_policy_seed,
        show_progress=True,
    )
    log_benchmark_results(
        output,
        config,
        config_name,
        splits,
        write_perfect_foresight_trajectories=args.write_perfect_foresight_trajectories,
        include_oracle_cloned_policy=args.include_oracle_cloned_policy,
        write_final_episode_metrics=args.write_final_episode_metrics,
        timeline_total_timesteps=timeline_total_timesteps,
        timeline_eval_freq=timeline_eval_freq,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
