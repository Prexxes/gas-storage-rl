"""Train one SB3 agent and evaluate it on validation/test splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from gas_storage_rl.agents.sb3_factory import make_sb3_agent
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.logging.experiment_logger import ExperimentLogger
from gas_storage_rl.logging.progress import CliProgress
from gas_storage_rl.training.callbacks import TrainingLoggingCallback
from gas_storage_rl.training.config import (
    build_effective_run_config,
    build_environment,
    load_config,
)


def load_pretrained_policy_state(model: Any, pretrained_policy: str | Path) -> Path:
    """Loads pretrained SB3 policy weights into a freshly created model."""
    policy_path = Path(pretrained_policy)
    if policy_path.is_dir():
        policy_path = policy_path / "policy_state_dict.pt"
    if not policy_path.exists():
        raise FileNotFoundError(f"Pretrained policy not found: {policy_path}")
    state_dict = torch.load(policy_path, map_location=model.device)
    model.policy.load_state_dict(state_dict)
    return policy_path


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
    args = parser.parse_args()

    config = load_config(args.config)
    config["agent_config"]["algorithm_name"] = args.algorithm
    progress = CliProgress("run_experiment", total=4)
    progress.step(1, "building environment")
    dataset, storage_params, env_kwargs = build_environment(config)
    train_env = GasStorageEnv(dataset, "train", storage_params, **env_kwargs)
    eval_env = GasStorageEnv(dataset, "validation", storage_params, **env_kwargs)
    effective_config = build_effective_run_config(config, args.algorithm)
    logger = ExperimentLogger(config["logging_config"]["run_dir"], effective_config)
    metadata = logger.metadata()
    logger.write_json("metadata.json", metadata)

    started = time.time()
    progress.step(2, "creating agent")
    model = make_sb3_agent(
        args.algorithm,
        train_env,
        config["agent_config"].get(args.algorithm, {}),
        seed=config["seeds"]["agent_seed"],
    )
    pretrained_policy_path = None
    if args.pretrained_policy:
        pretrained_policy_path = load_pretrained_policy_state(
            model,
            args.pretrained_policy,
        )
    from stable_baselines3.common.logger import configure

    sb3_logger = configure(str(logger.run_dir / "sb3_logs"), ["csv"])
    model.set_logger(sb3_logger)
    total_timesteps = int(config["training_config"]["total_timesteps"])
    callback = TrainingLoggingCallback(
        experiment_logger=logger,
        eval_env=eval_env,
        eval_freq=int(config["training_config"]["eval_freq"]),
        algorithm_name=args.algorithm,
        deterministic=bool(config["evaluation_config"].get("deterministic", True)),
        total_timesteps=total_timesteps,
        progress=CliProgress("training", total=total_timesteps),
    )
    progress.step(3, f"training {total_timesteps} steps")
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
    validation_metrics["algorithm_name"] = args.algorithm
    if callback.last_validation_step != total_timesteps:
        logger.append_csv("evaluations.csv", validation_metrics)
    summary = {
        "algorithm_name": args.algorithm,
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
    print(json.dumps({"run_dir": str(logger.run_dir), **summary}, indent=2))

if __name__ == "__main__":
    main()
