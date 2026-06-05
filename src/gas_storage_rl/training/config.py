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
        "seed": int(seeds.get("env_seed", 0)),
    }
    return dataset, storage_params, env_kwargs


def _build_price_process_config(config: dict[str, Any]) -> dict[str, Any]:
    """Returns fallback or historically calibrated price-process parameters."""
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
