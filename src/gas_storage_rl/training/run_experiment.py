"""Train one SB3 agent and evaluate it on validation/test splits."""

from __future__ import annotations

import argparse
import json
import time

from gas_storage_rl.agents.sb3_factory import make_sb3_agent
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.logging.experiment_logger import ExperimentLogger
from gas_storage_rl.training.callbacks import TrainingLoggingCallback
from gas_storage_rl.training.config import build_environment, load_config


def main() -> None:
    """Runs a configured experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    args = parser.parse_args()

    config = load_config(args.config)
    config["agent_config"]["algorithm_name"] = args.algorithm
    dataset, storage_params, env_kwargs = build_environment(config)
    train_env = GasStorageEnv(dataset, "train", storage_params, **env_kwargs)
    eval_env = GasStorageEnv(dataset, "validation", storage_params, **env_kwargs)
    test_env = GasStorageEnv(dataset, "test", storage_params, **env_kwargs)
    logger = ExperimentLogger(config["logging_config"]["run_dir"], config)
    logger.write_json("metadata.json", logger.metadata())

    started = time.time()
    model = make_sb3_agent(
        args.algorithm,
        train_env,
        config["agent_config"].get(args.algorithm, {}),
        seed=config["seeds"]["agent_seed"],
    )
    from stable_baselines3.common.logger import configure

    sb3_logger = configure(str(logger.run_dir / "sb3_logs"), ["csv"])
    model.set_logger(sb3_logger)
    callback = TrainingLoggingCallback(
        experiment_logger=logger,
        eval_env=eval_env,
        eval_freq=int(config["training_config"]["eval_freq"]),
        algorithm_name=args.algorithm,
        deterministic=bool(config["evaluation_config"].get("deterministic", True)),
    )
    total_timesteps = int(config["training_config"]["total_timesteps"])
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_wall_time = time.time() - started

    eval_started = time.time()
    validation_metrics, _ = evaluate_policy_on_paths(
        eval_env,
        model,
        deterministic=True,
        total_training_env_steps=total_timesteps,
    )
    test_metrics, _ = evaluate_policy_on_paths(
        test_env,
        model,
        deterministic=True,
        total_training_env_steps=total_timesteps,
    )
    eval_wall_time = time.time() - eval_started
    validation_metrics["algorithm_name"] = args.algorithm
    test_metrics["algorithm_name"] = args.algorithm
    if callback.last_validation_step != total_timesteps:
        logger.append_csv("evaluations.csv", validation_metrics)
    logger.append_csv("evaluations.csv", test_metrics)
    summary = {
        "algorithm_name": args.algorithm,
        "validation": validation_metrics,
        "test": test_metrics,
        "train_wall_time_s": train_wall_time,
        "eval_wall_time_s": eval_wall_time,
        "total_wall_time_s": time.time() - started,
    }
    logger.write_json("final_summary.json", summary)
    model.save(logger.run_dir / "final_model")
    print(json.dumps({"run_dir": str(logger.run_dir), **summary}, indent=2))

if __name__ == "__main__":
    main()
