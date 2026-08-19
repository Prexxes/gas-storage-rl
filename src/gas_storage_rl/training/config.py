"""YAML configuration loading and environment construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gas_storage_rl.data.path_dataset import PathDataset, load_or_generate_price_dataset
from gas_storage_rl.envs.storage_dynamics import StorageParams
from gas_storage_rl.prices.generators import PriceGeneratorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file.
    
    Args:
        path: Filesystem path to read from or write to.
    
    Returns:
        Parsed YAML configuration dictionary.

    """
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def seeds_for_run(seeds: dict[str, Any], seed_index: int) -> dict[str, int]:
    """Returns reproducible per-run seeds while preserving dataset/eval seeds.
    
    Args:
        seeds: Seeds value.
        seed_index: Zero-based seed repetition index.
    
    Returns:
        Deterministic seeds for the requested run index.
    
    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    if seed_index < 0:
        raise ValueError("seed_index must be non-negative")
    output = {key: int(value) for key, value in seeds.items()}
    sequence = np.random.SeedSequence([int(seeds["master_seed"]), seed_index])
    env_sequence, agent_sequence = sequence.spawn(2)
    output["env_seed"] = int(env_sequence.generate_state(1, dtype=np.uint32)[0])
    output["agent_seed"] = int(
        agent_sequence.generate_state(1, dtype=np.uint32)[0]
    )
    return output


def build_effective_run_config(
    config: dict[str, Any],
    algorithm_name: str,
    seed_index: int | None = None,
) -> dict[str, Any]:
    """Returns the configuration values that affect one training run.
    
    Args:
        config: Experiment configuration dictionary.
        algorithm_name: Algorithm name value.
        seed_index: Zero-based seed repetition index.
    
    Returns:
        Effective run configuration dictionary.

    """
    effective = {
        "environment_config": dict(config["environment_config"]),
        "dataset_config": _effective_dataset_config(config["dataset_config"]),
        "price_process_config": _effective_price_process_config(config),
        "backtest_data_config": _effective_backtest_data_config(config),
        "training_config": _effective_training_config(
            config["training_config"],
            seed_index,
        ),
        "evaluation_config": {
            "deterministic": bool(
                config.get("evaluation_config", {}).get("deterministic", True)
            ),
            "risk_adjusted_std_penalty": float(
                config.get("evaluation_config", {}).get(
                    "risk_adjusted_std_penalty",
                    0.5,
                )
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
    """Builds dataset, storage parameters, and environment keyword arguments.
    
    Args:
        config: Experiment configuration dictionary.
    
    Returns:
        Dataset, storage parameters, and environment keyword arguments.

    """
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
        "initial_inventory_mean_fraction": float(
            env_config.get("initial_inventory_mean_fraction", 0.30)
        ),
        "initial_inventory_std_fraction": float(
            env_config.get("initial_inventory_std_fraction", 0.05)
        ),
        "seed": int(seeds.get("env_seed", 0)),
    }
    if "observation_features" in env_config:
        env_kwargs["observation_features"] = dict(
            env_config["observation_features"]
        )
    return dataset, storage_params, env_kwargs


def _build_price_process_config(config: dict[str, Any]) -> dict[str, Any]:
    """Returns configured synthetic price-process parameters.
    
    Args:
        config: Experiment configuration dictionary.
    
    Returns:
        Build price process config result.

    """
    price_process_config = config.get("price_process_config", {})
    if "parameters" in price_process_config:
        price_config = dict(price_process_config["parameters"])
        price_config["_environment_name"] = price_process_config.get(
            "environment_name",
            config["environment_config"]["environment_name"],
        )
        return price_config
    return dict(price_process_config)


def _effective_price_process_config(config: dict[str, Any]) -> dict[str, Any]:
    """Returns logged price-process provenance and effective parameters.
    
    Args:
        config: Experiment configuration dictionary.
    
    Returns:
        Effective price process config result.

    """
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
    return effective


def _effective_backtest_data_config(config: dict[str, Any]) -> dict[str, Any]:
    """Returns historical backtest settings used by holdout evaluations.
    
    Args:
        config: Experiment configuration dictionary.
    
    Returns:
        Backtest data configuration, or an empty dictionary when absent.

    """
    backtest_config = config.get("backtest_data_config")
    if backtest_config is None:
        return {}
    keys = (
        "daily_backtest_csv",
        "backtest_start_date",
        "daily_backtest_price_column",
        "window_stride",
    )
    return {key: backtest_config[key] for key in keys if key in backtest_config}


def _effective_dataset_config(dataset_config: dict[str, Any]) -> dict[str, Any]:
    """Returns dataset settings used by training runs.
    
    Args:
        dataset_config: Dataset config value.
    
    Returns:
        Effective dataset config result.

    """
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
    """Returns training settings used by one run.
    
    Args:
        training_config: Training configuration dictionary.
        seed_index: Zero-based seed repetition index.
    
    Returns:
        Effective training config result.

    """
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
    """Returns non-plot seeds used by training and evaluation.
    
    Args:
        seeds: Seeds value.
    
    Returns:
        Effective seeds result.

    """
    keys = ("master_seed", "dataset_seed", "env_seed", "agent_seed", "eval_seed")
    return {key: int(seeds[key]) for key in keys if key in seeds}
