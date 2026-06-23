"""Synthetic gas spot price dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gas_storage_rl.prices.calibration import default_price_parameters
from gas_storage_rl.prices.jump_process import simulate_jump_component
from gas_storage_rl.prices.ou_process import simulate_additive_ou_process
from gas_storage_rl.prices.seasonal import (
    evaluate_fourier_monthly_seasonality,
)


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
        """Validates rolling-window and inventory-distribution parameters."""
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
        """Returns the stored raw-path length used for rolling episodes."""
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
    """Generates synthetic or historically calibrated spot price paths.

    Args:
        environment_name: One of ``deterministic``, ``ou``, ``jump``,
            ``historical_deterministic``, ``historical_ou``,
            ``historical_jump``, or ``historical_calibrated``.
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

    historical_environment_names = {
        "historical_deterministic",
        "historical_ou",
        "historical_jump",
        "historical_calibrated",
    }
    if environment_name in historical_environment_names:
        return generate_calibrated_price_paths(
            n_paths=n_paths,
            episode_length=episode_length,
            seed=seed,
            params=process_params,
            include_ou=environment_name
            in {"historical_ou", "historical_jump", "historical_calibrated"},
            include_jumps=environment_name
            in {"historical_jump", "historical_calibrated"},
        )

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
    """Generates additive synthetic price paths that may become negative."""
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


def generate_calibrated_price_paths(
    n_paths: int,
    episode_length: int,
    seed: int,
    params: dict[str, Any],
    include_ou: bool = True,
    include_jumps: bool = True,
) -> np.ndarray:
    """Generates paths from calibrated monthly seasonality, AR(1), and jumps.

    Args:
        n_paths: Number of paths.
        episode_length: Number of days.
        seed: Random seed.
        params: Calibrated price process parameters.
        include_ou: Whether to add calibrated AR(1)/OU residual noise.
        include_jumps: Whether to add calibrated sparse jumps.

    Returns:
        Positive price matrix with shape ``(n_paths, episode_length)``.
    """
    rng = np.random.default_rng(seed)
    seasonal = calibrated_monthly_log_curve(
        episode_length=episode_length,
        base_log_price=float(params["base_log_price"]),
        monthly_log_seasonality=params["monthly_log_seasonality"],
        method=str(params.get("seasonality_interpolation", "fourier")),
        fourier_harmonics=int(params.get("seasonality_fourier_harmonics", 2)),
    )
    residuals = np.zeros((n_paths, episode_length), dtype=np.float64)
    if include_ou:
        residuals = _simulate_ar1_residuals(
            n_paths=n_paths,
            episode_length=episode_length,
            phi=float(
                params.get("ar1_phi", 1.0 - params.get("ou_mean_reversion", 0.0))
            ),
            volatility=float(params.get("ou_volatility", 0.0)),
            rng=rng,
        )
    jumps = np.zeros((n_paths, episode_length), dtype=np.float64)
    if include_jumps:
        jumps = _simulate_calibrated_jumps(
            n_paths=n_paths,
            episode_length=episode_length,
            jump_probability=float(params.get("jump_probability", 0.0)),
            jump_mean=float(params.get("jump_mean", 0.0)),
            jump_std=float(params.get("jump_std", 0.0)),
            rng=rng,
        )
    prices = np.exp(np.tile(seasonal, (n_paths, 1)) + residuals + jumps)
    return np.maximum(prices, np.finfo(np.float64).tiny).astype(np.float32)


def calibrated_monthly_log_curve(
    episode_length: int,
    base_log_price: float,
    monthly_log_seasonality: list[float] | tuple[float, ...],
    start_date: str = "2001-01-01",
    method: str = "fourier",
    fourier_harmonics: int = 2,
) -> np.ndarray:
    """Creates a daily log seasonal curve from calibrated monthly values."""
    if len(monthly_log_seasonality) != 12:
        raise ValueError("monthly_log_seasonality must contain exactly 12 values")
    dates = pd.date_range(start=start_date, periods=episode_length, freq="D")
    adjustments = np.asarray(monthly_log_seasonality, dtype=np.float64)
    if method == "step":
        return base_log_price + adjustments[dates.month.to_numpy() - 1]
    if method != "fourier":
        raise ValueError(f"Unknown seasonality method: {method}")
    return base_log_price + evaluate_fourier_monthly_seasonality(
        dates,
        adjustments,
        harmonics=fourier_harmonics,
    )


def _simulate_ar1_residuals(
    n_paths: int,
    episode_length: int,
    phi: float,
    volatility: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulates centered AR(1) residuals."""
    residuals = np.zeros((n_paths, episode_length), dtype=np.float64)
    if episode_length <= 1 or volatility <= 0.0:
        return residuals
    shocks = rng.normal(0.0, volatility, size=(n_paths, episode_length - 1))
    for step in range(1, episode_length):
        residuals[:, step] = phi * residuals[:, step - 1] + shocks[:, step - 1]
    return residuals


def _simulate_calibrated_jumps(
    n_paths: int,
    episode_length: int,
    jump_probability: float,
    jump_mean: float,
    jump_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulates calibrated sparse additive log-price jumps."""
    jumps = np.zeros((n_paths, episode_length), dtype=np.float64)
    if jump_probability <= 0.0:
        return jumps
    jump_mask = rng.random((n_paths, episode_length)) < jump_probability
    if jump_std > 0.0:
        jump_sizes = rng.normal(jump_mean, jump_std, size=(n_paths, episode_length))
    else:
        jump_sizes = np.full((n_paths, episode_length), jump_mean, dtype=np.float64)
    return jump_mask * jump_sizes


def split_seeds(dataset_seed: int, include_pretrain: bool = False) -> dict[str, int]:
    """Returns disjoint deterministic seeds for dataset splits."""
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
    """Generates pretrain, train, validation, and test price paths."""
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
