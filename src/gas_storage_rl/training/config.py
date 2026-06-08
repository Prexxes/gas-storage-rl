"""YAML configuration loading and environment construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gas_storage_rl.data.path_dataset import PathDataset, load_or_generate_price_dataset
from gas_storage_rl.envs.storage_dynamics import StorageParams
from gas_storage_rl.prices.calibration import calibrate_historical_price_process
from gas_storage_rl.prices.generators import PriceGeneratorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_effective_run_config(
    config: dict[str, Any],
    algorithm_name: str,
    seed_index: int | None = None,
) -> dict[str, Any]:
    """Returns the configuration values that affect one training run."""
    effective = {
        "environment_config": dict(config["environment_config"]),
        "dataset_config": _effective_dataset_config(config["dataset_config"]),
        "price_process_config": _effective_price_process_config(config),
        "training_config": _effective_training_config(
            config["training_config"],
            seed_index,
        ),
        "evaluation_config": {
            "deterministic": bool(
                config.get("evaluation_config", {}).get("deterministic", True)
            ),
            "evaluation_split": "validation",
        },
        "seeds": _effective_seeds(config["seeds"]),
        "agent_config": {
            "algorithm_name": algorithm_name,
            "hyperparameters": dict(config["agent_config"].get(algorithm_name, {})),
        },
        "logging_config": {
            "run_dir": config.get("logging_config", {}).get("run_dir", "runs"),
        },
    }
    return effective


def build_environment(
    config: dict[str, Any],
) -> tuple[PathDataset, StorageParams, dict[str, Any]]:
    """Builds dataset, storage parameters, and environment keyword arguments."""
    env_config = config["environment_config"]
    dataset_config = config["dataset_config"]
    price_config = _build_price_process_config(config)
    seeds = config["seeds"]
    environment_name = price_config.pop(
        "_environment_name",
        env_config["environment_name"],
    )
    price_generator_config = PriceGeneratorConfig(
        environment_name=environment_name,
        episode_length=env_config["episode_length"],
        n_pretrain_paths=int(dataset_config.get("n_pretrain_paths", 0)),
        n_train_paths=dataset_config["n_train_paths"],
        n_validation_paths=dataset_config["n_validation_paths"],
        n_test_paths=dataset_config["n_test_paths"],
        dataset_seed=seeds["dataset_seed"],
        max_start_offset=int(
            dataset_config.get(
                "max_start_offset",
                env_config["episode_length"] - 1,
            )
        ),
        storage_capacity=float(env_config["capacity"]),
        initial_inventory_mean_fraction=float(
            env_config.get("initial_inventory_mean_fraction", 0.30)
        ),
        initial_inventory_std_fraction=float(
            env_config.get("initial_inventory_std_fraction", 0.05)
        ),
        params=price_config,
    )
    dataset = load_or_generate_price_dataset(
        price_generator_config,
        cache_dir=dataset_config.get("cache_dir", "data/cache"),
        use_cache=bool(dataset_config.get("use_cache", True)),
        force_regenerate=bool(dataset_config.get("force_regenerate", False)),
    )
    storage_params = StorageParams(
        capacity=float(env_config["capacity"]),
        injection_rate=float(env_config.get("injection_rate", 1.0)),
        withdrawal_rate=float(env_config.get("withdrawal_rate", 1.0)),
        initial_inventory=float(env_config.get("initial_inventory", 0.0)),
        target_terminal_inventory=float(
            env_config.get("target_terminal_inventory", 0.0)
        ),
        efficiency=float(env_config.get("efficiency", 1.0)),
        transaction_cost=float(env_config.get("transaction_cost", 0.0)),
        leakage=float(env_config.get("leakage", 0.0)),
    )
    env_kwargs = {
        "price_scale": float(
            env_config.get("price_scale", price_config.get("base_price", 50.0))
        ),
        "reward_scale": float(
            env_config.get("reward_scale", price_config.get("base_price", 50.0))
        ),
        "penalty_factor": float(env_config.get("penalty_factor", 0.5)),
        "feasibility_penalty_factor": float(
            env_config.get("feasibility_penalty_factor", 0.5)
        ),
        "initial_inventory_mean_fraction": float(
            env_config.get("initial_inventory_mean_fraction", 0.30)
        ),
        "initial_inventory_std_fraction": float(
            env_config.get("initial_inventory_std_fraction", 0.05)
        ),
        "seed": int(seeds.get("env_seed", 0)),
    }
    return dataset, storage_params, env_kwargs


def _build_price_process_config(config: dict[str, Any]) -> dict[str, Any]:
    """Returns fallback or historically calibrated price-process parameters."""
    price_process_config = config.get("price_process_config", {})
    if "parameters" in price_process_config:
        price_config = dict(price_process_config["parameters"])
        price_config["_environment_name"] = price_process_config.get(
            "environment_name",
            config["environment_config"]["environment_name"],
        )
        return price_config

    calibrated_config = config.get("calibrated_price_process_config", {})
    if not bool(calibrated_config.get("enabled", False)):
        return dict(config["price_process_config"])

    historical_config = config["historical_data_config"]
    calibration = calibrate_historical_price_process(
        historical_config["monthly_calibration_csv"],
        historical_config["daily_calibration_csv"],
        calibration_end_date=historical_config.get(
            "calibration_end_date",
            "2024-12-31",
        ),
        monthly_price_column=historical_config.get("monthly_price_column"),
        daily_price_column=historical_config.get("daily_calibration_price_column"),
        jump_threshold_sigma=float(calibrated_config.get("jump_threshold_sigma", 3.0)),
    )
    price_config = dict(calibration.to_price_params())
    price_config["_environment_name"] = calibrated_config.get(
        "environment_name",
        "historical_calibrated",
    )
    return price_config


def _effective_price_process_config(config: dict[str, Any]) -> dict[str, Any]:
    """Returns logged price-process provenance and effective parameters."""
    price_config = _build_price_process_config(config)
    environment_name = price_config.pop(
        "_environment_name",
        config["environment_config"]["environment_name"],
    )
    effective = {
        "source": "config",
        "environment_name": environment_name,
        "parameters": price_config,
    }
    calibrated_config = config.get("calibrated_price_process_config", {})
    if bool(calibrated_config.get("enabled", False)):
        historical_config = config["historical_data_config"]
        effective["source"] = "historical_calibration"
        effective["calibration"] = {
            "monthly_calibration_csv": historical_config["monthly_calibration_csv"],
            "daily_calibration_csv": historical_config["daily_calibration_csv"],
            "calibration_end_date": historical_config.get(
                "calibration_end_date",
                "2024-12-31",
            ),
            "jump_threshold_sigma": float(
                calibrated_config.get("jump_threshold_sigma", 3.0)
            ),
        }
        if historical_config.get("monthly_price_column") is not None:
            effective["calibration"]["monthly_price_column"] = historical_config[
                "monthly_price_column"
            ]
        if historical_config.get("daily_calibration_price_column") is not None:
            effective["calibration"]["daily_price_column"] = historical_config[
                "daily_calibration_price_column"
            ]
    return effective


def _effective_dataset_config(dataset_config: dict[str, Any]) -> dict[str, Any]:
    """Returns dataset settings used by training runs."""
    keys = (
        "n_pretrain_paths",
        "n_train_paths",
        "n_validation_paths",
        "n_test_paths",
        "cache_dir",
        "use_cache",
        "force_regenerate",
        "max_start_offset",
    )
    return {key: dataset_config[key] for key in keys if key in dataset_config}


def _effective_training_config(
    training_config: dict[str, Any],
    seed_index: int | None,
) -> dict[str, Any]:
    """Returns training settings used by one run."""
    effective = {
        "total_timesteps": int(training_config["total_timesteps"]),
        "eval_freq": int(training_config["eval_freq"]),
    }
    if seed_index is not None:
        effective["seed_index"] = int(seed_index)
    if "n_seeds" in training_config:
        effective["n_seeds"] = int(training_config["n_seeds"])
    return effective


def _effective_seeds(seeds: dict[str, Any]) -> dict[str, int]:
    """Returns non-plot seeds used by training and evaluation."""
    keys = ("master_seed", "dataset_seed", "env_seed", "agent_seed", "eval_seed")
    return {key: int(seeds[key]) for key in keys if key in seeds}
