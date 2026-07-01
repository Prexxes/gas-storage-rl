"""Phase-1 Optuna hyperparameter tuning for RL algorithms."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from gas_storage_rl.hpo.search_spaces import (
    search_space_description,
    suggest_hyperparameters,
)
from gas_storage_rl.training.config import load_config, seeds_for_run
from gas_storage_rl.training.run_experiment import run_experiment

DEFAULT_HPO_SEED_INDICES = [0, 1, 2]
DEFAULT_N_TRIALS = 32
DEFAULT_TOTAL_TIMESTEPS = 500_000


def run_hpo(
    config: dict[str, Any],
    config_name: str,
    algorithm: str,
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    seed_indices: list[int] | None = None,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    study_name: str | None = None,
    rerun: bool = False,
) -> dict[str, Any]:
    """Runs phase-1 HPO using Optuna TPE and returns a study summary."""
    import optuna

    seed_indices = list(DEFAULT_HPO_SEED_INDICES if seed_indices is None else seed_indices)
    _validate_hpo_inputs(algorithm, n_trials, seed_indices)
    study_dir, study_id = _create_study_dir(config, config_name, algorithm, study_name)
    storage_url = f"sqlite:///{study_dir / 'optuna_study.db'}"
    hpo_config = _hpo_config(
        config,
        config_name,
        algorithm,
        study_id,
        seed_indices,
        n_trials,
        total_timesteps,
    )
    _write_json(study_dir / "study_config.json", hpo_config)
    _write_json(study_dir / "search_space.json", search_space_description())
    _write_json(
        study_dir / "metadata.json",
        {
            "study_id": study_id,
            "config_name": config_name,
            "algorithm_name": algorithm,
            "hpo_method": "Optuna TPE",
            "storage": storage_url,
            "direction": "maximize",
            "n_trials": n_trials,
            "seed_indices": seed_indices,
            "total_timesteps": total_timesteps,
            "objective": "mean_validation_return_raw",
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    sampler = optuna.samplers.TPESampler(seed=int(config["seeds"]["master_seed"]))
    study = optuna.create_study(
        study_name=study_id,
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    trial_rows: list[dict[str, Any]] = []
    seed_run_rows: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        return _run_trial(
            trial,
            config,
            algorithm,
            study_id,
            seed_indices,
            total_timesteps,
            rerun,
            trial_rows,
            seed_run_rows,
            study_dir,
        )

    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))
    best_config = _best_config(config, algorithm, total_timesteps, study.best_params)
    best_config["agent_config"][algorithm] = suggest_hyperparameters(
        _FixedTrial(study.best_trial.params),
        algorithm,
    )
    _write_json(study_dir / "best_config.json", best_config)
    _write_json(
        study_dir / "best_trial.json",
        {
            "trial_id": int(study.best_trial.number),
            "objective_mean_validation_return_raw": float(study.best_value),
            "hyperparameters": best_config["agent_config"][algorithm],
            "params": dict(study.best_trial.params),
        },
    )
    metadata = _read_json(study_dir / "metadata.json")
    metadata["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    metadata["best_trial_id"] = int(study.best_trial.number)
    metadata["best_value"] = float(study.best_value)
    _write_json(study_dir / "metadata.json", metadata)
    return {
        "study_dir": str(study_dir),
        "study_id": study_id,
        "best_trial_id": int(study.best_trial.number),
        "best_value": float(study.best_value),
    }


def _run_trial(
    trial: Any,
    config: dict[str, Any],
    algorithm: str,
    study_id: str,
    seed_indices: list[int],
    total_timesteps: int,
    rerun: bool,
    trial_rows: list[dict[str, Any]],
    seed_run_rows: list[dict[str, Any]],
    study_dir: Path,
) -> float:
    """Runs all seed repetitions for one Optuna trial."""
    started = time.time()
    hyperparameters = suggest_hyperparameters(trial, algorithm)
    validation_values = []
    seed_rows_for_trial = []
    status = "completed"
    error_message = ""
    for seed_index in seed_indices:
        row = _run_trial_seed(
            config,
            algorithm,
            study_id,
            int(trial.number),
            seed_index,
            hyperparameters,
            total_timesteps,
            rerun,
        )
        seed_run_rows.append(row)
        seed_rows_for_trial.append(row)
        _write_csv(study_dir / "trial_seed_runs.csv", seed_run_rows)
        if row["status"] != "completed" and row["status"] != "skipped":
            status = "failed"
            error_message = row["error"]
            break
        validation_values.append(float(row["mean_validation_return_raw"]))

    if len(validation_values) != len(seed_indices):
        trial_row = _trial_row(
            algorithm,
            int(trial.number),
            status,
            hyperparameters,
            total_timesteps,
            time.time() - started,
            [],
            error_message,
        )
        trial_rows.append(trial_row)
        _write_csv(study_dir / "trials.csv", trial_rows)
        raise RuntimeError(
            f"Trial {trial.number} failed before all seed runs completed: "
            f"{error_message}"
        )

    objective = float(statistics.mean(validation_values))
    trial_row = _trial_row(
        algorithm,
        int(trial.number),
        status,
        hyperparameters,
        total_timesteps,
        time.time() - started,
        validation_values,
        error_message,
    )
    trial_rows.append(trial_row)
    _write_csv(study_dir / "trials.csv", trial_rows)
    _write_json(study_dir / f"trial_{int(trial.number):04d}.json", {
        "trial_id": int(trial.number),
        "algorithm": algorithm,
        "status": status,
        "hyperparameters": hyperparameters,
        "seed_runs": seed_rows_for_trial,
        "objective_mean_validation_return_raw": objective,
    })
    return objective


def _run_trial_seed(
    config: dict[str, Any],
    algorithm: str,
    study_id: str,
    trial_id: int,
    seed_index: int,
    hyperparameters: dict[str, Any],
    total_timesteps: int,
    rerun: bool,
) -> dict[str, Any]:
    """Runs one trial/seed training run and returns a flat result row."""
    run_config = copy.deepcopy(config)
    run_config["training_config"]["total_timesteps"] = int(total_timesteps)
    run_config["agent_config"][algorithm] = copy.deepcopy(hyperparameters)
    run_config["seeds"] = seeds_for_run(config["seeds"], seed_index)
    started = time.time()
    row = {
        "algorithm": algorithm,
        "trial_id": trial_id,
        "seed_index": seed_index,
        "tuning_seed": seed_index,
        "dataset_seed": int(run_config["seeds"]["dataset_seed"]),
        "env_seed": int(run_config["seeds"]["env_seed"]),
        "agent_seed": int(run_config["seeds"]["agent_seed"]),
        "eval_seed": int(run_config["seeds"]["eval_seed"]),
        "run_dir": "",
        "mean_validation_return_raw": "",
        "std_validation_return_raw": "",
        "median_validation_return_raw": "",
        "min_validation_return_raw": "",
        "terminal_deviation": "",
        "clipped_action_count": "",
        "total_timesteps": int(total_timesteps),
        "runtime_s": "",
        "status": "running",
        "error": "",
    }
    try:
        summary = run_experiment(
            run_config,
            algorithm,
            seed_index=seed_index,
            experiment_group_id=f"{study_id}-trial-{trial_id:04d}",
            rerun=rerun,
        )
        metrics = summary["validation"]
        row.update(
            {
                "run_dir": summary["run_dir"],
                "mean_validation_return_raw": float(metrics["mean_return_raw"]),
                "std_validation_return_raw": float(metrics["std_return_raw"]),
                "median_validation_return_raw": float(metrics["median_return_raw"]),
                "min_validation_return_raw": float(metrics["min_return_raw"]),
                "terminal_deviation": float(metrics["mean_terminal_deviation"]),
                "clipped_action_count": float(
                    metrics["mean_number_of_constrained_actions"]
                ),
                "runtime_s": float(time.time() - started),
                "status": summary.get("status", "completed"),
            }
        )
    except Exception as error:
        row["runtime_s"] = float(time.time() - started)
        row["status"] = "failed"
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def _trial_row(
    algorithm: str,
    trial_id: int,
    status: str,
    hyperparameters: dict[str, Any],
    total_timesteps: int,
    runtime_s: float,
    validation_values: list[float],
    error_message: str,
) -> dict[str, Any]:
    """Returns one flat trial-level CSV row."""
    row = {
        "algorithm": algorithm,
        "trial_id": trial_id,
        "status": status,
        "objective_mean_validation_return_raw": "",
        "std_validation_return_raw_across_seeds": "",
        "median_validation_return_raw_across_seeds": "",
        "min_validation_return_raw_across_seeds": "",
        "total_timesteps": int(total_timesteps),
        "runtime_s": float(runtime_s),
        "hyperparameters_json": json.dumps(hyperparameters, sort_keys=True),
        "error": error_message,
    }
    if validation_values:
        row.update(
            {
                "objective_mean_validation_return_raw": float(
                    statistics.mean(validation_values)
                ),
                "std_validation_return_raw_across_seeds": float(
                    statistics.pstdev(validation_values)
                ),
                "median_validation_return_raw_across_seeds": float(
                    statistics.median(validation_values)
                ),
                "min_validation_return_raw_across_seeds": float(
                    min(validation_values)
                ),
            }
        )
    return row


def _best_config(
    config: dict[str, Any],
    algorithm: str,
    total_timesteps: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Returns a config copy with fixed total timesteps and algorithm marker."""
    del params
    output = copy.deepcopy(config)
    output["training_config"]["total_timesteps"] = int(total_timesteps)
    output["agent_config"]["algorithm_name"] = algorithm
    return output


def _hpo_config(
    config: dict[str, Any],
    config_name: str,
    algorithm: str,
    study_id: str,
    seed_indices: list[int],
    n_trials: int,
    total_timesteps: int,
) -> dict[str, Any]:
    """Returns the saved HPO configuration."""
    return {
        "study_id": study_id,
        "config_name": config_name,
        "algorithm_name": algorithm,
        "hpo_method": "Optuna TPE",
        "n_trials": n_trials,
        "tuning_seed_indices": seed_indices,
        "total_timesteps": total_timesteps,
        "objective": "mean_validation_return_raw",
        "direction": "maximize",
        "phase": "phase_1_hyperparameter_tuning",
        "train_split": "train",
        "selection_split": "validation",
        "uses_test_split": False,
        "base_config": config,
    }


def _create_study_dir(
    config: dict[str, Any],
    config_name: str,
    algorithm: str,
    study_name: str | None,
) -> tuple[Path, str]:
    """Creates a unique HPO study directory."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if study_name:
        study_id = _safe_component(study_name)
    else:
        study_id = f"{timestamp}-{_safe_component(config_name)}-{algorithm}"
    base_dir = Path(config["logging_config"]["run_dir"]) / "hpo"
    study_dir = base_dir / study_id
    suffix = 1
    while study_dir.exists():
        study_id = f"{study_id}-{suffix}"
        study_dir = base_dir / study_id
        suffix += 1
    study_dir.mkdir(parents=True, exist_ok=False)
    return study_dir, study_id


def _validate_hpo_inputs(
    algorithm: str,
    n_trials: int,
    seed_indices: list[int],
) -> None:
    """Validates HPO runner settings."""
    if algorithm not in {"ppo", "sac", "td3"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if len(seed_indices) != len(set(seed_indices)):
        raise ValueError("seed_indices must be unique")
    if not seed_indices:
        raise ValueError("seed_indices must not be empty")


def _safe_component(value: Any) -> str:
    """Returns a filesystem-safe id component."""
    text = str(value)
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in text
    )
    return cleaned or "study"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes one JSON document."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def _read_json(path: Path) -> dict[str, Any]:
    """Reads one JSON document."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes rows to a CSV file."""
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class _FixedTrial:
    """Small Optuna FixedTrial-compatible adapter for best-config export."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initializes the adapter from stored Optuna params."""
        self.params = params

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        """Returns a fixed categorical value."""
        value = self.params[name]
        if value not in choices:
            raise ValueError(f"Fixed value for {name} is not in choices: {value}")
        return value

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
    ) -> float:
        """Returns a fixed float value."""
        del low, high, log
        return float(self.params[name])


def main() -> None:
    """Runs phase-1 Optuna HPO from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument(
        "--seed-indices",
        type=int,
        nargs="+",
        default=DEFAULT_HPO_SEED_INDICES,
    )
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--study-name")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run even if completed training runs with matching configs exist.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    summary = run_hpo(
        load_config(config_path),
        config_path.stem,
        args.algorithm,
        n_trials=args.n_trials,
        seed_indices=args.seed_indices,
        total_timesteps=args.total_timesteps,
        study_name=args.study_name,
        rerun=args.rerun,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
