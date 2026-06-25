"""Run a sequential group of RL experiments across reproducible seeds."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from gas_storage_rl.training.config import load_config, seeds_for_run
from gas_storage_rl.training.run_experiment import run_experiment


def run_experiment_group(
    config: dict[str, Any],
    config_name: str,
    algorithm: str,
    *,
    n_seeds: int | None = None,
    pretrained_policy: str | Path | None = None,
    rerun: bool = False,
) -> dict[str, Any]:
    """Runs one experiment per seed index and records a group manifest."""
    run_count = int(
        config["training_config"].get("n_seeds", 1)
        if n_seeds is None
        else n_seeds
    )
    if run_count <= 0:
        raise ValueError("n_seeds must be positive")

    group_dir, group_id = _create_group_dir(config, config_name, algorithm)
    _write_json(group_dir / "group_config.json", config)
    started = time.time()
    rows = []
    for seed_index in range(run_count):
        run_config = copy.deepcopy(config)
        run_config["seeds"] = seeds_for_run(config["seeds"], seed_index)
        row = {
            "experiment_group_id": group_id,
            "algorithm_name": algorithm,
            "seed_index": seed_index,
            "master_seed": int(run_config["seeds"]["master_seed"]),
            "dataset_seed": int(run_config["seeds"]["dataset_seed"]),
            "eval_seed": int(run_config["seeds"]["eval_seed"]),
            "env_seed": int(run_config["seeds"]["env_seed"]),
            "agent_seed": int(run_config["seeds"]["agent_seed"]),
            "status": "running",
            "run_dir": "",
            "error": "",
        }
        rows.append(row)
        _write_csv(group_dir / "runs.csv", rows)
        try:
            summary = run_experiment(
                run_config,
                algorithm,
                seed_index=seed_index,
                experiment_group_id=group_id,
                pretrained_policy=pretrained_policy,
                rerun=rerun,
            )
            row["status"] = summary.get("status", "completed")
            row["run_dir"] = summary["run_dir"]
        except Exception as error:  # Continue so remaining seeds still run.
            row["status"] = "failed"
            row["error"] = f"{type(error).__name__}: {error}"
        _write_csv(group_dir / "runs.csv", rows)

    completed = sum(row["status"] == "completed" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    metadata = {
        "experiment_group_id": group_id,
        "config_name": config_name,
        "algorithm_name": algorithm,
        "n_seeds": run_count,
        "completed_runs": completed,
        "skipped_runs": skipped,
        "failed_runs": failed,
        "dataset_seed": int(config["seeds"]["dataset_seed"]),
        "eval_seed": int(config["seeds"]["eval_seed"]),
        "total_wall_time_s": time.time() - started,
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_json(group_dir / "group_metadata.json", metadata)
    return {"group_dir": str(group_dir), **metadata, "runs": rows}


def _create_group_dir(
    config: dict[str, Any],
    config_name: str,
    algorithm: str,
) -> tuple[Path, str]:
    """Creates a unique experiment-group directory."""
    payload = json.dumps(config, sort_keys=True, default=str)
    hash_value = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path(config["logging_config"]["run_dir"]) / "experiment_groups"
    group_id = f"{timestamp}-{config_name}-{algorithm}-{hash_value}"
    group_dir = base_dir / group_id
    suffix = 1
    while group_dir.exists():
        group_id = f"{timestamp}-{config_name}-{algorithm}-{hash_value}-{suffix}"
        group_dir = base_dir / group_id
        suffix += 1
    group_dir.mkdir(parents=True, exist_ok=False)
    return group_dir, group_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes one JSON document."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes the current experiment-group manifest."""
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Runs a configured sequential experiment group."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--pretrained-policy")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run seeds even if completed runs with the same effective configs exist.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    summary = run_experiment_group(
        load_config(config_path),
        config_path.stem,
        args.algorithm,
        n_seeds=args.n_seeds,
        pretrained_policy=args.pretrained_policy,
        rerun=args.rerun,
    )
    print(json.dumps(summary, indent=2))
    if summary["failed_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
