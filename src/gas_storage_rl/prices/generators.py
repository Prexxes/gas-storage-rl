"""Synthetic gas spot price dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gas_storage_rl.prices.jump_process import simulate_jump_component
from gas_storage_rl.prices.ou_process import simulate_additive_ou_process
from gas_storage_rl.prices.parameters import default_price_parameters


@dataclass(frozen=True)
class PriceGeneratorConfig:
    """Configuration for synthetic spot price generation."""

    environment_name: str
    episode_length: int = 365
    n_pretrain_paths: int = 0
    n_train_paths: int = 5000
    n_validation_paths: int = 500
    n_test_paths: int = 1000
    dataset_seed: int = 123
    max_start_offset: int | None = None
    storage_capacity: float = 1.0
    initial_inventory_mean_fraction: float = 0.30
    initial_inventory_std_fraction: float = 0.05
    params: dict[str, Any] = field(default_factory=default_price_parameters)

    def __post_init__(self) -> None:
        """Validates rolling-window and inventory-distribution parameters.
        
        Raises:
            ValueError: If an input value or configuration is invalid.

        """
        if self.storage_capacity <= 0.0:
            raise ValueError("storage_capacity must be positive")
        if not 0.0 <= self.initial_inventory_mean_fraction <= 1.0:
            raise ValueError("initial_inventory_mean_fraction must be in [0, 1]")
        if self.initial_inventory_std_fraction < 0.0:
            raise ValueError("initial_inventory_std_fraction must be non-negative")
        if self.max_start_offset is not None and self.max_start_offset < 0:
            raise ValueError("max_start_offset must be non-negative")

    @property
    def simulation_length(self) -> int:
        """Returns the stored raw-path length used for rolling episodes.
        
        Returns:
            Number of simulated time steps including rollout padding.

        """
        max_start_offset = (
            self.episode_length - 1
            if self.max_start_offset is None
            else self.max_start_offset
        )
        return self.episode_length + max_start_offset


def generate_price_paths(
    environment_name: str,
    n_paths: int,
    episode_length: int,
    seed: int,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Generates synthetic spot price paths.
    
    Args:
        environment_name: One of ``deterministic``, ``ou``, or ``jump``.
        n_paths: Number of paths.
        episode_length: Number of days.
        seed: Random seed.
        params: Optional process parameters.
    
    Returns:
        Positive price matrix with shape ``(n_paths, episode_length)``.
    
    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    process_params = default_price_parameters()
    if params:
        process_params.update(params)

    if environment_name not in {"deterministic", "ou", "jump"}:
        raise ValueError(f"Unknown environment_name: {environment_name}")
    return generate_additive_price_paths(
        environment_name=environment_name,
        n_paths=n_paths,
        episode_length=episode_length,
        seed=seed,
        params=process_params,
    )


def generate_additive_price_paths(
    environment_name: str,
    n_paths: int,
    episode_length: int,
    seed: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Generates additive synthetic price paths that may become negative.
    
    Args:
        environment_name: Environment name value.
        n_paths: Number of price paths to generate or evaluate.
        episode_length: Episode length value.
        seed: Random seed for deterministic behavior.
        params: Params value.
    
    Returns:
        Computed result.

    """
    days = np.arange(episode_length, dtype=np.float64)
    seasonal = params["seasonal_level"] + params["seasonal_amplitude"] * np.sin(
        2.0 * np.pi * days / params["seasonal_period"]
    )
    prices = np.tile(seasonal, (n_paths, 1))
    if environment_name in {"ou", "jump"}:
        prices += simulate_additive_ou_process(
            n_paths=n_paths,
            episode_length=episode_length,
            speed_of_mean_reversion=params["ou_speed_of_mean_reversion"],
            long_term_mean=params["ou_long_term_mean"],
            volatility=params["ou_volatility"],
            start_value=params["ou_start_value"],
            time_step=params["ou_time_step"],
            seed=seed,
        )
    if environment_name == "jump":
        prices += simulate_jump_component(
            n_paths=n_paths,
            episode_length=episode_length,
            jump_probability=params["jump_probability"],
            jump_mean=params["jump_mean"],
            jump_std=params["jump_std"],
            stress_probability=params["stress_probability"],
            stress_multiplier=params["stress_multiplier"],
            seed=seed + 10_000,
        )
    return prices.astype(np.float32)


def split_seeds(dataset_seed: int, include_pretrain: bool = False) -> dict[str, int]:
    """Returns disjoint deterministic seeds for dataset splits.
    
    Args:
        dataset_seed: Dataset seed value.
        include_pretrain: Include pretrain value.
    
    Returns:
        Deterministic seed mapping by split.

    """
    seeds = {
        "train": int(dataset_seed),
        "validation": int(dataset_seed + 1_000_003),
        "test": int(dataset_seed + 2_000_003),
    }
    if include_pretrain:
        seeds = {
            "pretrain": int(dataset_seed + 3_000_003),
            **seeds,
        }
    return seeds


def generate_dataset(config: PriceGeneratorConfig) -> dict[str, np.ndarray]:
    """Generates pretrain, train, validation, and test price paths.
    
    Args:
        config: Experiment configuration dictionary.
    
    Returns:
        Generated price path dataset by split.

    """
    include_pretrain = config.n_pretrain_paths > 0
    seeds = split_seeds(config.dataset_seed, include_pretrain=include_pretrain)
    counts = {
        "train": config.n_train_paths,
        "validation": config.n_validation_paths,
        "test": config.n_test_paths,
    }
    if include_pretrain:
        counts = {"pretrain": config.n_pretrain_paths, **counts}
    return {
        split: generate_price_paths(
            environment_name=config.environment_name,
            n_paths=count,
            episode_length=config.simulation_length,
            seed=seeds[split],
            params=config.params,
        )
        for split, count in counts.items()
    }
