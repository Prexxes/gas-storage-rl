"""Synthetic gas spot price dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gas_storage_rl.prices.calibration import default_price_parameters
from gas_storage_rl.prices.jump_process import simulate_jump_component
from gas_storage_rl.prices.ou_process import simulate_ou_residuals
from gas_storage_rl.prices.seasonal import seasonal_log_curve


@dataclass(frozen=True)
class PriceGeneratorConfig:
    """Configuration for synthetic spot price generation."""

    environment_name: str
    episode_length: int = 365
    n_train_paths: int = 5000
    n_validation_paths: int = 500
    n_test_paths: int = 1000
    dataset_seed: int = 123
    params: dict[str, float] = field(default_factory=default_price_parameters)


def generate_price_paths(
    environment_name: str,
    n_paths: int,
    episode_length: int,
    seed: int,
    params: dict[str, float] | None = None,
) -> np.ndarray:
    """Generates strictly positive synthetic spot price paths.

    Args:
        environment_name: One of ``deterministic``, ``ou``, or ``jump``.
        n_paths: Number of paths.
        episode_length: Number of days.
        seed: Random seed.
        params: Optional process parameters.

    Returns:
        Positive price matrix with shape ``(n_paths, episode_length)``.
    """
    process_params = default_price_parameters()
    if params:
        process_params.update(params)

    seasonal = seasonal_log_curve(
        episode_length=episode_length,
        base_price=process_params["base_price"],
        amplitude=process_params["seasonal_amplitude"],
        phase=process_params["seasonal_phase"],
    )
    log_prices = np.tile(seasonal, (n_paths, 1))

    if environment_name in {"ou", "jump"}:
        log_prices += simulate_ou_residuals(
            n_paths=n_paths,
            episode_length=episode_length,
            mean_reversion=process_params["ou_mean_reversion"],
            volatility=process_params["ou_volatility"],
            seed=seed,
        )
    if environment_name == "jump":
        log_prices += simulate_jump_component(
            n_paths=n_paths,
            episode_length=episode_length,
            jump_probability=process_params["jump_probability"],
            jump_mean=process_params["jump_mean"],
            jump_std=process_params["jump_std"],
            stress_probability=process_params["stress_probability"],
            stress_multiplier=process_params["stress_multiplier"],
            seed=seed + 10_000,
        )
    if environment_name not in {"deterministic", "ou", "jump"}:
        raise ValueError(f"Unknown environment_name: {environment_name}")

    prices = np.exp(log_prices)
    return np.maximum(prices, np.finfo(np.float64).tiny).astype(np.float32)


def split_seeds(dataset_seed: int) -> dict[str, int]:
    """Returns disjoint deterministic seeds for dataset splits."""
    return {
        "train": int(dataset_seed),
        "validation": int(dataset_seed + 1_000_003),
        "test": int(dataset_seed + 2_000_003),
    }


def generate_dataset(config: PriceGeneratorConfig) -> dict[str, np.ndarray]:
    """Generates train, validation, and test price paths."""
    seeds = split_seeds(config.dataset_seed)
    counts = {
        "train": config.n_train_paths,
        "validation": config.n_validation_paths,
        "test": config.n_test_paths,
    }
    return {
        split: generate_price_paths(
            environment_name=config.environment_name,
            n_paths=count,
            episode_length=config.episode_length,
            seed=seeds[split],
            params=config.params,
        )
        for split, count in counts.items()
    }
