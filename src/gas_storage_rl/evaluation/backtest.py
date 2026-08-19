"""Helpers for evaluating policies on held-out historical backtest windows."""

from __future__ import annotations

from typing import Any

import numpy as np

from gas_storage_rl.data.path_dataset import (
    PathDataset,
    load_or_generate_historical_backtest_dataset,
)
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.envs.storage_dynamics import StorageParams
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.training.config import build_environment


def build_backtest_evaluation_dataset(
    config: dict[str, Any],
    synthetic_dataset: PathDataset,
) -> PathDataset:
    """Returns a dataset with synthetic splits plus held-out backtest windows.

    Args:
        config: Experiment configuration dictionary.
        synthetic_dataset: Existing synthetic train/validation/test dataset.

    Returns:
        Dataset containing all synthetic splits and a ``backtest`` split.

    """
    backtest_config = _backtest_data_config(config)
    dataset_config = config["dataset_config"]
    backtest_dataset = load_or_generate_historical_backtest_dataset(
        backtest_config["daily_backtest_csv"],
        episode_length=int(config["environment_config"]["episode_length"]),
        cache_dir=dataset_config.get("backtest_cache_dir", "data/cache/backtest"),
        use_cache=bool(dataset_config.get("use_cache", True)),
        force_regenerate=bool(dataset_config.get("force_regenerate", False)),
        window_stride=int(backtest_config.get("window_stride", 1)),
        backtest_start_date=backtest_config.get(
            "backtest_start_date",
            "2025-01-01",
        ),
        daily_price_column=backtest_config.get("daily_backtest_price_column"),
    )
    env_config = config["environment_config"]
    default_inventory = float(env_config.get("initial_inventory", 0.0))
    initial_inventories_by_split = {
        split: synthetic_dataset.get_initial_inventories(split, default_inventory)
        for split in synthetic_dataset.paths_by_split
    }
    initial_inventories_by_split["backtest"] = _sample_backtest_initial_inventories(
        config,
        n_paths=len(backtest_dataset.get_paths("backtest")),
    )

    start_indices_by_split = None
    if synthetic_dataset.start_indices_by_split is not None:
        start_indices_by_split = dict(synthetic_dataset.start_indices_by_split)
        start_indices_by_split["backtest"] = np.zeros(
            len(backtest_dataset.get_paths("backtest")),
            dtype=np.int64,
        )

    date_ranges_by_split = dict(synthetic_dataset.date_ranges_by_split or {})
    date_ranges_by_split["backtest"] = backtest_dataset.date_ranges_by_split[
        "backtest"
    ]
    paths_by_split = dict(synthetic_dataset.paths_by_split)
    paths_by_split["backtest"] = backtest_dataset.get_paths("backtest")
    seeds_by_split = dict(synthetic_dataset.seeds_by_split)
    seeds_by_split["backtest"] = 0
    base_dates_by_split = (
        dict(synthetic_dataset.base_dates_by_split)
        if synthetic_dataset.base_dates_by_split is not None
        else None
    )
    return PathDataset(
        paths_by_split,
        seeds_by_split,
        date_ranges_by_split=date_ranges_by_split,
        episode_length_override=synthetic_dataset.episode_length,
        start_indices_by_split=start_indices_by_split,
        base_dates_by_split=base_dates_by_split,
        initial_inventories_by_split=initial_inventories_by_split,
    )


def evaluate_policy_on_backtest(
    policy: Any,
    config: dict[str, Any],
    *,
    synthetic_dataset: PathDataset | None = None,
    storage_params: StorageParams | None = None,
    env_kwargs: dict[str, Any] | None = None,
    deterministic: bool = True,
    total_training_env_steps: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluates any SB3-style policy on historical backtest windows.

    Args:
        policy: Policy object exposing ``predict(observation, deterministic=...)``.
        config: Experiment configuration dictionary.
        synthetic_dataset: Optional prebuilt synthetic dataset.
        storage_params: Optional prebuilt storage parameters.
        env_kwargs: Optional prebuilt environment keyword arguments.
        deterministic: Whether to request deterministic policy actions.
        total_training_env_steps: Optional training-step metadata for metrics.

    Returns:
        Evaluation metrics and per-path trajectories.

    """
    if synthetic_dataset is None or storage_params is None or env_kwargs is None:
        synthetic_dataset, storage_params, env_kwargs = build_environment(config)
    dataset = build_backtest_evaluation_dataset(config, synthetic_dataset)
    env = GasStorageEnv(dataset, "backtest", storage_params, **env_kwargs)
    return evaluate_policy_on_paths(
        env,
        policy,
        deterministic=deterministic,
        total_training_env_steps=total_training_env_steps,
    )


def _backtest_data_config(config: dict[str, Any]) -> dict[str, Any]:
    """Returns configured held-out historical backtest settings."""
    backtest_config = config.get("backtest_data_config")
    if backtest_config and "daily_backtest_csv" in backtest_config:
        return backtest_config
    raise ValueError("backtest_data_config.daily_backtest_csv is required")


def _sample_backtest_initial_inventories(
    config: dict[str, Any],
    n_paths: int,
) -> np.ndarray:
    """Samples deterministic initial inventories for backtest episodes."""
    env_config = config["environment_config"]
    rng = np.random.default_rng(int(config["seeds"]["eval_seed"]) + 5_000_003)
    fractions = rng.normal(
        float(env_config.get("initial_inventory_mean_fraction", 0.30)),
        float(env_config.get("initial_inventory_std_fraction", 0.05)),
        size=n_paths,
    )
    return (np.clip(fractions, 0.0, 1.0) * float(env_config["capacity"])).astype(
        np.float64
    )
