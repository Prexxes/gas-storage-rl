"""Phase-1 Optuna hyperparameter tuning for RL algorithms."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from gas_storage_rl.hpo.search_spaces import (
    search_space_description,
    suggest_hyperparameters,
    suggest_reward_scale_multiplier,
)
from gas_storage_rl.training.config import load_config, seeds_for_run
from gas_storage_rl.training.run_experiment import run_experiment

DEFAULT_HPO_SEED_INDICES = [0, 1, 2]
DEFAULT_N_TRIALS = 32
DEFAULT_TOTAL_TIMESTEPS = 500_000
HPO_OBJECTIVE = "mean_across_seeds_of_max_validation_mean_return_raw"
HPO_OBJECTIVE_SELECTION = "max_mean_return_raw"


def run_hpo(
    config: dict[str, Any],
    config_name: str,
    algorithm: str,
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    seed_indices: list[int] | None = None,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    study_name: str | None = None,
    resume_study_dir: str | Path | None = None,
    rerun: bool = False,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Runs phase-1 HPO using Optuna TPE and returns a study summary."""
    import optuna

    seed_indices = list(DEFAULT_HPO_SEED_INDICES if seed_indices is None else seed_indices)
    _validate_hpo_inputs(algorithm, n_trials, seed_indices, n_jobs)
    study_dir, study_id, is_resume = _study_dir(
        config,
        config_name,
        algorithm,
        study_name,
        resume_study_dir,
    )
    storage_url = f"sqlite:///{study_dir / 'optuna_study.db'}?timeout=60"
    hpo_config = _hpo_config(
        config,
        config_name,
        algorithm,
        study_id,
        seed_indices,
        n_trials,
        total_timesteps,
        n_jobs,
    )
    _write_json(study_dir / "study_config.json", hpo_config)
    _write_json(study_dir / "search_space.json", search_space_description())

    sampler = optuna.samplers.TPESampler(seed=int(config["seeds"]["master_seed"]))
    study = optuna.create_study(
        study_name=study_id,
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )
    finished_trials_before = _finished_trial_count(study)
    remaining_trials = max(0, int(n_trials) - finished_trials_before)
    _write_metadata(
        study_dir,
        {
            "study_id": study_id,
            "config_name": config_name,
            "algorithm_name": algorithm,
            "hpo_method": "Optuna TPE",
            "storage": storage_url,
            "direction": "maximize",
            "n_trials": n_trials,
            "target_n_trials": n_trials,
            "n_existing_finished_trials": finished_trials_before,
            "n_remaining_trials": remaining_trials,
            "n_jobs": n_jobs,
            "seed_indices": seed_indices,
            "total_timesteps": total_timesteps,
            "objective": HPO_OBJECTIVE,
            "is_resume": is_resume,
        },
        is_resume,
    )

    def objective(trial: Any) -> float:
        return _run_trial(
            trial,
            config,
            algorithm,
            study_id,
            seed_indices,
            total_timesteps,
            rerun,
            study_dir,
        )

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=n_jobs,
            catch=(RuntimeError,),
        )
    _export_trial_csvs(study_dir)
    best_config = _best_config(config, algorithm, total_timesteps, study.best_params)
    best_hyperparameters = suggest_hyperparameters(
        _FixedTrial(study.best_trial.params),
        algorithm,
    )
    best_config["agent_config"][algorithm] = _agent_config_with_hyperparameters(
        config,
        algorithm,
        best_hyperparameters,
    )
    best_reward_scale_info = _reward_scale_info(config, study.best_trial.params)
    _write_json(study_dir / "best_config.json", best_config)
    _write_json(
        study_dir / "best_trial.json",
        {
            "trial_id": int(study.best_trial.number),
            "objective_mean_validation_return_raw": float(study.best_value),
            "hyperparameters": best_config["agent_config"][algorithm],
            "reward_scale_multiplier": best_reward_scale_info[
                "reward_scale_multiplier"
            ],
            "base_reward_scale": best_reward_scale_info["base_reward_scale"],
            "effective_reward_scale": best_reward_scale_info[
                "effective_reward_scale"
            ],
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
        "target_n_trials": int(n_trials),
        "n_existing_finished_trials": finished_trials_before,
        "n_remaining_trials": remaining_trials,
    }


def _run_trial(
    trial: Any,
    config: dict[str, Any],
    algorithm: str,
    study_id: str,
    seed_indices: list[int],
    total_timesteps: int,
    rerun: bool,
    study_dir: Path,
) -> float:
    """Runs all seed repetitions for one Optuna trial."""
    started = time.time()
    reward_scale_multiplier = suggest_reward_scale_multiplier(trial)
    reward_scale_info = _reward_scale_info(
        config,
        {"reward_scale_multiplier": reward_scale_multiplier},
    )
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
            reward_scale_multiplier,
            total_timesteps,
            rerun,
        )
        seed_rows_for_trial.append(row)
        if row["status"] != "completed" and row["status"] != "skipped":
            status = "failed"
            error_message = row["error"]
            break
        validation_values.append(float(row["hpo_objective_mean_validation_return_raw"]))

    if len(validation_values) != len(seed_indices):
        trial_row = _trial_row(
            algorithm,
            int(trial.number),
            status,
            hyperparameters,
            reward_scale_info,
            total_timesteps,
            time.time() - started,
            [],
            error_message,
        )
        _write_trial_payload(
            study_dir,
            int(trial.number),
            algorithm,
            status,
            hyperparameters,
            reward_scale_info,
            seed_rows_for_trial,
            trial_row,
            None,
        )
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
        reward_scale_info,
        total_timesteps,
        time.time() - started,
        validation_values,
        error_message,
    )
    _write_trial_payload(
        study_dir,
        int(trial.number),
        algorithm,
        status,
        hyperparameters,
        reward_scale_info,
        seed_rows_for_trial,
        trial_row,
        objective,
    )
    return objective


def _write_trial_payload(
    study_dir: Path,
    trial_id: int,
    algorithm: str,
    status: str,
    hyperparameters: dict[str, Any],
    reward_scale_info: dict[str, float],
    seed_rows_for_trial: list[dict[str, Any]],
    trial_row: dict[str, Any],
    objective: float | None,
) -> None:
    """Writes one trial-specific artifact for parallel-safe HPO exports."""
    payload = {
        "trial_id": trial_id,
        "algorithm": algorithm,
        "status": status,
        "hyperparameters": hyperparameters,
        "reward_scale_multiplier": reward_scale_info["reward_scale_multiplier"],
        "base_reward_scale": reward_scale_info["base_reward_scale"],
        "effective_reward_scale": reward_scale_info["effective_reward_scale"],
        "seed_runs": seed_rows_for_trial,
        "trial_row": trial_row,
        "objective_mean_validation_return_raw": "" if objective is None else objective,
    }
    _write_json_atomic(study_dir / f"trial_{trial_id:04d}.json", payload)


def _run_trial_seed(
    config: dict[str, Any],
    algorithm: str,
    study_id: str,
    trial_id: int,
    seed_index: int,
    hyperparameters: dict[str, Any],
    reward_scale_multiplier: float,
    total_timesteps: int,
    rerun: bool,
) -> dict[str, Any]:
    """Runs one trial/seed training run and returns a flat result row."""
    run_config = copy.deepcopy(config)
    run_config["training_config"]["total_timesteps"] = int(total_timesteps)
    run_config["agent_config"][algorithm] = _agent_config_with_hyperparameters(
        config,
        algorithm,
        hyperparameters,
    )
    reward_scale_info = _apply_reward_scale_multiplier(
        run_config,
        reward_scale_multiplier,
    )
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
        "reward_scale_multiplier": reward_scale_info["reward_scale_multiplier"],
        "base_reward_scale": reward_scale_info["base_reward_scale"],
        "effective_reward_scale": reward_scale_info["effective_reward_scale"],
        "run_dir": "",
        "mean_validation_return_raw": "",
        "std_validation_return_raw": "",
        "median_validation_return_raw": "",
        "min_validation_return_raw": "",
        "terminal_deviation": "",
        "clipped_action_count": "",
        "total_timesteps": int(total_timesteps),
        "runtime_s": "",
        "hpo_objective_selection": "",
        "hpo_objective_training_env_steps": "",
        "hpo_objective_mean_validation_return_raw": "",
        "final_mean_validation_return_raw": "",
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
        final_metrics = summary["validation"]
        metrics = _select_hpo_objective_metrics(summary)
        row.update(
            {
                "run_dir": summary["run_dir"],
                "hpo_objective_selection": metrics["hpo_objective_selection"],
                "hpo_objective_training_env_steps": metrics.get(
                    "total_training_env_steps",
                    "",
                ),
                "hpo_objective_mean_validation_return_raw": float(
                    metrics["mean_return_raw"]
                ),
                "final_mean_validation_return_raw": float(
                    final_metrics["mean_return_raw"]
                ),
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


def _select_hpo_objective_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Returns the validation row used for HPO objective selection."""
    rows = _read_evaluation_rows(Path(summary["run_dir"]))
    if not rows:
        metrics = dict(summary["validation"])
        metrics["hpo_objective_selection"] = "final_mean_return_raw"
        return metrics

    best_row = max(rows, key=lambda row: float(row["mean_return_raw"]))
    metrics = _coerce_evaluation_row(best_row)
    metrics["hpo_objective_selection"] = HPO_OBJECTIVE_SELECTION
    return metrics


def _read_evaluation_rows(run_dir: Path) -> list[dict[str, str]]:
    """Reads validation rows for one training run."""
    path = run_dir / "evaluations.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _coerce_evaluation_row(row: dict[str, str]) -> dict[str, Any]:
    """Converts numeric CSV evaluation values to floats."""
    output: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            output[key] = value
            continue
        try:
            output[key] = float(value)
        except ValueError:
            output[key] = value
    return output


def _trial_row(
    algorithm: str,
    trial_id: int,
    status: str,
    hyperparameters: dict[str, Any],
    reward_scale_info: dict[str, float],
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
        "reward_scale_multiplier": reward_scale_info["reward_scale_multiplier"],
        "base_reward_scale": reward_scale_info["base_reward_scale"],
        "effective_reward_scale": reward_scale_info["effective_reward_scale"],
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
    output = copy.deepcopy(config)
    output["training_config"]["total_timesteps"] = int(total_timesteps)
    output["agent_config"]["algorithm_name"] = algorithm
    _apply_reward_scale_multiplier(
        output,
        float(params["reward_scale_multiplier"]),
    )
    return output


def _agent_config_with_hyperparameters(
    config: dict[str, Any],
    algorithm: str,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Merges tuned hyperparameters into the base algorithm config."""
    agent_config = copy.deepcopy(config.get("agent_config", {}).get(algorithm, {}))
    agent_config.update(copy.deepcopy(hyperparameters))
    return agent_config


def _base_reward_scale(config: dict[str, Any]) -> float:
    """Returns the base reward scale using the environment defaulting rules."""
    environment_config = config.get("environment_config", {})
    price_config = config.get("price_process_config", {})
    return float(
        environment_config.get(
            "reward_scale",
            price_config.get("base_price", 50.0),
        )
    )


def _reward_scale_info(
    config: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, float]:
    """Returns base, multiplier, and effective reward-scale values."""
    multiplier = float(params["reward_scale_multiplier"])
    base_reward_scale = _base_reward_scale(config)
    return {
        "reward_scale_multiplier": multiplier,
        "base_reward_scale": base_reward_scale,
        "effective_reward_scale": base_reward_scale * multiplier,
    }


def _apply_reward_scale_multiplier(
    config: dict[str, Any],
    reward_scale_multiplier: float,
) -> dict[str, float]:
    """Applies the reward-scale multiplier to a config copy."""
    reward_scale_info = _reward_scale_info(
        config,
        {"reward_scale_multiplier": reward_scale_multiplier},
    )
    config.setdefault("environment_config", {})["reward_scale"] = reward_scale_info[
        "effective_reward_scale"
    ]
    return reward_scale_info


def _hpo_config(
    config: dict[str, Any],
    config_name: str,
    algorithm: str,
    study_id: str,
    seed_indices: list[int],
    n_trials: int,
    total_timesteps: int,
    n_jobs: int,
) -> dict[str, Any]:
    """Returns the saved HPO configuration."""
    return {
        "study_id": study_id,
        "config_name": config_name,
        "algorithm_name": algorithm,
        "hpo_method": "Optuna TPE",
        "n_trials": n_trials,
        "n_jobs": n_jobs,
        "tuning_seed_indices": seed_indices,
        "total_timesteps": total_timesteps,
        "objective": HPO_OBJECTIVE,
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


def _study_dir(
    config: dict[str, Any],
    config_name: str,
    algorithm: str,
    study_name: str | None,
    resume_study_dir: str | Path | None,
) -> tuple[Path, str, bool]:
    """Returns the HPO study directory, optionally resuming an existing one."""
    if resume_study_dir is None:
        study_dir, study_id = _create_study_dir(
            config,
            config_name,
            algorithm,
            study_name,
        )
        return study_dir, study_id, False

    if study_name is not None:
        raise ValueError("study_name must not be set when resume_study_dir is used")
    study_dir = Path(resume_study_dir)
    if not study_dir.exists() or not study_dir.is_dir():
        raise FileNotFoundError(f"resume_study_dir not found: {study_dir}")
    if not (study_dir / "optuna_study.db").exists():
        raise FileNotFoundError(
            f"resume_study_dir does not contain optuna_study.db: {study_dir}"
        )
    metadata_path = study_dir / "metadata.json"
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
        study_id = str(metadata.get("study_id", study_dir.name))
    else:
        study_id = study_dir.name
    return study_dir, study_id, True


def _finished_trial_count(study: Any) -> int:
    """Returns the number of Optuna trials in a finished state."""
    return sum(1 for trial in study.trials if trial.state.is_finished())


def _validate_hpo_inputs(
    algorithm: str,
    n_trials: int,
    seed_indices: list[int],
    n_jobs: int = 1,
) -> None:
    """Validates HPO runner settings."""
    if algorithm not in {"ppo", "sac", "td3"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")
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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Writes one JSON document through a same-directory atomic replace."""
    temp_path = path.with_name(
        f".{path.name}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    _write_json(temp_path, payload)
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Reads one JSON document."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_metadata(
    study_dir: Path,
    payload: dict[str, Any],
    is_resume: bool,
) -> None:
    """Writes or updates HPO metadata."""
    path = study_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if is_resume and path.exists():
        metadata = _read_json(path)
    else:
        metadata["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    metadata.update(payload)
    if is_resume:
        metadata["last_resume_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(path, metadata)


def _trial_payload_paths(study_dir: Path) -> list[Path]:
    """Returns completed per-trial payload paths in deterministic order."""
    return sorted(study_dir.glob("trial_[0-9][0-9][0-9][0-9].json"))


def _read_trial_payloads(study_dir: Path) -> list[dict[str, Any]]:
    """Reads completed per-trial payloads in trial-id order."""
    payloads = [_read_json(path) for path in _trial_payload_paths(study_dir)]
    payloads.sort(key=lambda payload: int(payload["trial_id"]))
    return payloads


def _export_trial_csvs(study_dir: Path) -> None:
    """Exports aggregate HPO CSVs from per-trial JSON artifacts."""
    payloads = _read_trial_payloads(study_dir)
    trial_rows = [payload["trial_row"] for payload in payloads]
    seed_rows = [
        row
        for payload in payloads
        for row in payload.get("seed_runs", [])
    ]
    seed_rows.sort(key=lambda row: (int(row["trial_id"]), int(row["seed_index"])))
    _write_csv(study_dir / "trials.csv", trial_rows)
    _write_csv(study_dir / "trial_seed_runs.csv", seed_rows)


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
        "--n-jobs",
        type=int,
        default=1,
        help="Number of Optuna trials to run in parallel.",
    )
    parser.add_argument(
        "--seed-indices",
        type=int,
        nargs="+",
        default=DEFAULT_HPO_SEED_INDICES,
    )
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--study-name")
    parser.add_argument(
        "--resume-study-dir",
        help=(
            "Existing runs/hpo/<study_id> directory to continue. In resume mode, "
            "--n-trials is interpreted as the target total number of finished "
            "Optuna trials."
        ),
    )
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
        resume_study_dir=args.resume_study_dir,
        rerun=args.rerun,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
