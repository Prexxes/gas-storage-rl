"""YAML configuration loading and environment construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gas_storage_rl.data.path_dataset import PathDataset, load_or_generate_price_dataset
from gas_storage_rl.envs.storage_dynamics import StorageParams
from gas_storage_rl.prices.generators import PriceGeneratorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_environment(config: dict[str, Any]) -> tuple[PathDataset, StorageParams, dict[str, Any]]:
    """Builds dataset, storage parameters, and environment keyword arguments."""
    env_config = config["environment_config"]
    dataset_config = config["dataset_config"]
    price_config = config["price_process_config"]
    seeds = config["seeds"]
    price_generator_config = PriceGeneratorConfig(
        environment_name=env_config["environment_name"],
        episode_length=env_config["episode_length"],
        n_train_paths=dataset_config["n_train_paths"],
        n_validation_paths=dataset_config["n_validation_paths"],
        n_test_paths=dataset_config["n_test_paths"],
        dataset_seed=seeds["dataset_seed"],
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
        target_terminal_inventory=float(env_config.get("target_terminal_inventory", 0.0)),
        efficiency=float(env_config.get("efficiency", 1.0)),
        transaction_cost=float(env_config.get("transaction_cost", 0.0)),
        leakage=float(env_config.get("leakage", 0.0)),
    )
    env_kwargs = {
        "price_scale": float(env_config.get("price_scale", price_config.get("base_price", 50.0))),
        "reward_scale": float(env_config.get("reward_scale", price_config.get("base_price", 50.0))),
        "penalty_factor": float(env_config.get("penalty_factor", 0.5)),
        "seed": int(seeds.get("env_seed", 0)),
    }
    return dataset, storage_params, env_kwargs
