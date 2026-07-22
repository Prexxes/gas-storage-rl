"""Train one SB3 agent and evaluate it on validation/test splits."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from gas_storage_rl.agents.sb3_factory import (
    effective_sb3_hyperparameters,
    make_sb3_agent,
)
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.evaluation.metrics import (
    add_risk_adjusted_return,
    validation_return_aulc,
)
from gas_storage_rl.logging.experiment_logger import ExperimentLogger
from gas_storage_rl.logging.progress import CliProgress
from gas_storage_rl.training.callbacks import TrainingLoggingCallback
from gas_storage_rl.training.config import (
    build_effective_run_config,
    build_environment,
    load_config,
)


def _json_default(value: Any) -> str:
    """JSON fallback serializer for run fingerprints.
    
    Args:
        value: Value value.
    
    Returns:
        Json default result.

    """
    return str(value)


def run_config_fingerprint(config: dict[str, Any]) -> str:
    """Returns a stable fingerprint for a rerun-relevant run configuration.
    
    Args:
        config: Experiment configuration dictionary.
    
    Returns:
        Computed result.

    """
    normalized = copy.deepcopy(config)
    normalized.pop("experiment_group_id", None)
    normalized.pop("logging_config", None)
    normalized.get("agent_config", {}).pop("effective_hyperparameters", None)
    serialized = json.dumps(
        normalized,
        sort_keys=True,
        default=_json_default,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def find_completed_run(
    base_dir: str | Path,
    effective_config: dict[str, Any],
) -> Path | None:
    """Finds a completed run with the same rerun-relevant configuration.
    
    Args:
        base_dir: Parent directory for generated artifacts.
        effective_config: Effective config value.
    
    Returns:
        Computed result.

    """
    run_base_dir = Path(base_dir)
    if not run_base_dir.exists():
        return None

    target_fingerprint = run_config_fingerprint(effective_config)
    for run_dir in sorted(run_base_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        final_summary_path = run_dir / "final_summary.json"
        if not config_path.exists() or not final_summary_path.exists():
            continue
        try:
            with config_path.open("r", encoding="utf-8") as file:
                candidate_config = json.load(file)
        except json.JSONDecodeError:
            continue
        if run_config_fingerprint(candidate_config) == target_fingerprint:
            return run_dir
    return None


def pretrained_policy_reference(pretrained_policy: str | Path) -> str:
    """Returns the effective policy file reference used by a training run.
    
    Args:
        pretrained_policy: Pretrained policy value.
    
    Returns:
        Computed result.

    """
    policy_path = Path(pretrained_policy)
    if policy_path.is_dir():
        policy_path = policy_path / "policy_state_dict.pt"
    return str(policy_path)


def load_pretrained_policy_state(model: Any, pretrained_policy: str | Path) -> Path:
    """Loads pretrained SB3 policy weights into a freshly created model.
    
    Args:
        model: Policy or model used for prediction.
        pretrained_policy: Pretrained policy value.
    
    Returns:
        Computed result.
    
    Raises:
        FileNotFoundError: If a required input file does not exist.

    """
    policy_path = Path(pretrained_policy_reference(pretrained_policy))
    if not policy_path.exists():
        raise FileNotFoundError(f"Pretrained policy not found: {policy_path}")
    state_dict = torch.load(policy_path, map_location=model.device)
    model.policy.load_state_dict(state_dict)
    return policy_path


def _read_evaluation_rows(run_dir: Path) -> list[dict[str, str]]:
    """Reads periodic and final validation rows for one run.
    
    Args:
        run_dir: Run dir value.
    
    Returns:
        Read evaluation rows result.

    """
    path = run_dir / "evaluations.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def run_experiment(
    config: dict[str, Any],
    algorithm: str,
    *,
    seed_index: int | None = None,
    experiment_group_id: str | None = None,
    pretrained_policy: str | Path | None = None,
    rerun: bool = False,
) -> dict[str, Any]:
    """Runs one configured RL experiment and returns its summary.
    
    Args:
        config: Experiment configuration dictionary.
        algorithm: Algorithm name to configure or evaluate.
        seed_index: Zero-based seed repetition index.
        experiment_group_id: Experiment group id value.
        pretrained_policy: Pretrained policy value.
        rerun: Rerun value.
    
    Returns:
        Run summary and artifact metadata.

    """
    config["agent_config"]["algorithm_name"] = algorithm
    effective_config = build_effective_run_config(config, algorithm, seed_index)
    if pretrained_policy is not None:
        effective_config["pretrained_policy"] = pretrained_policy_reference(
            pretrained_policy
        )
    if not rerun:
        completed_run = find_completed_run(
            config["logging_config"]["run_dir"],
            effective_config,
        )
        if completed_run is not None:
            with (completed_run / "final_summary.json").open(
                "r",
                encoding="utf-8",
            ) as file:
                summary = json.load(file)
            return {
                **summary,
                "run_dir": str(completed_run),
                "status": "skipped",
                "existing_run_dir": str(completed_run),
            }

    progress = CliProgress("run_experiment", total=4)
    progress.step(1, "building environment")
    dataset, storage_params, env_kwargs = build_environment(config)
    train_env = GasStorageEnv(dataset, "train", storage_params, **env_kwargs)
    eval_env = GasStorageEnv(dataset, "validation", storage_params, **env_kwargs)
    if experiment_group_id is not None:
        effective_config["experiment_group_id"] = experiment_group_id
    logger = ExperimentLogger(config["logging_config"]["run_dir"], effective_config)
    metadata = logger.metadata()
    metadata["seed_index"] = seed_index
    metadata["experiment_group_id"] = experiment_group_id
    logger.write_json("metadata.json", metadata)

    started = time.time()
    progress.step(2, "creating agent")
    model = make_sb3_agent(
        algorithm,
        train_env,
        config["agent_config"].get(algorithm, {}),
        seed=config["seeds"]["agent_seed"],
    )
    effective_config["agent_config"]["effective_hyperparameters"] = (
        effective_sb3_hyperparameters(
            algorithm,
            config["agent_config"].get(algorithm, {}),
            seed=config["seeds"]["agent_seed"],
        )
    )
    logger.config = effective_config
    logger.write_json("config.json", effective_config)
    pretrained_policy_path = None
    if pretrained_policy:
        pretrained_policy_path = load_pretrained_policy_state(
            model,
            pretrained_policy,
        )
    from stable_baselines3.common.logger import configure

    sb3_logger = configure(str(logger.run_dir / "sb3_logs"), ["csv"])
    model.set_logger(sb3_logger)
    total_timesteps = int(config["training_config"]["total_timesteps"])
    callback = TrainingLoggingCallback(
        experiment_logger=logger,
        eval_env=eval_env,
        eval_freq=int(config["training_config"]["eval_freq"]),
        algorithm_name=algorithm,
        deterministic=bool(config["evaluation_config"].get("deterministic", True)),
        risk_adjusted_std_penalty=float(
            config.get("evaluation_config", {}).get("risk_adjusted_std_penalty", 0.5)
        ),
        total_timesteps=total_timesteps,
        progress=CliProgress("training", total=total_timesteps),
    )
    progress.step(3, f"training {algorithm} for {total_timesteps} steps")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_wall_time = time.time() - started

    eval_started = time.time()
    progress.step(4, "final validation")
    validation_metrics, _ = evaluate_policy_on_paths(
        eval_env,
        model,
        deterministic=True,
        total_training_env_steps=total_timesteps,
    )
    eval_wall_time = time.time() - eval_started
    validation_metrics["algorithm_name"] = algorithm
    add_risk_adjusted_return(
        validation_metrics,
        float(config.get("evaluation_config", {}).get("risk_adjusted_std_penalty", 0.5)),
    )
    logger.append_csv("evaluations.csv", validation_metrics)
    validation_metrics.update(
        validation_return_aulc(
            _read_evaluation_rows(logger.run_dir),
            total_timesteps,
        )
    )
    summary = {
        "status": "completed",
        "algorithm_name": algorithm,
        "seed_index": seed_index,
        "experiment_group_id": experiment_group_id,
        "agent_seed": int(config["seeds"]["agent_seed"]),
        "env_seed": int(config["seeds"]["env_seed"]),
        "dataset_seed": int(config["seeds"]["dataset_seed"]),
        "eval_seed": int(config["seeds"]["eval_seed"]),
        "pretrained_policy": str(pretrained_policy_path)
        if pretrained_policy_path is not None
        else None,
        "validation": validation_metrics,
        "train_wall_time_s": train_wall_time,
        "eval_wall_time_s": eval_wall_time,
        "total_wall_time_s": time.time() - started,
    }
    logger.write_json("final_summary.json", summary)
    model.save(logger.run_dir / "final_model")
    logger.finalize_metadata(metadata)
    progress.finish("done")
    return {"run_dir": str(logger.run_dir), **summary}


def main() -> None:
    """Runs a configured experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument(
        "--pretrained-policy",
        help=(
            "Path to policy_state_dict.pt or a pretraining run directory. "
            "The RL model is still created from the current config."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run even if a completed run with the same effective config exists.",
    )
    args = parser.parse_args()

    summary = run_experiment(
        load_config(args.config),
        args.algorithm,
        pretrained_policy=args.pretrained_policy,
        rerun=args.rerun,
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
