"""Tests for effective training run configuration logging."""

from __future__ import annotations

from gas_storage_rl.training.config import (
    _build_price_process_config,
    build_effective_run_config,
)


def _base_config() -> dict:
    """Returns a config containing both used and unused run settings."""
    return {
        "environment_config": {
            "environment_name": "ou",
            "capacity": 30,
            "episode_length": 10,
        },
        "dataset_config": {
            "n_train_paths": 4,
            "n_validation_paths": 2,
            "n_test_paths": 2,
            "cache_dir": "data/cache",
            "backtest_cache_dir": "data/cache/backtest",
            "use_cache": True,
            "force_regenerate": False,
        },
        "training_config": {
            "total_timesteps": 32,
            "eval_freq": 16,
            "n_seeds": 10,
        },
        "evaluation_config": {
            "deterministic": True,
            "lsmc_action_grid": [-1.0, 0.0, 1.0],
        },
        "plotting_config": {
            "plot_split": "validation",
            "plot_n_paths": 50,
        },
        "seeds": {
            "master_seed": 1,
            "dataset_seed": 2,
            "env_seed": 3,
            "agent_seed": 4,
            "eval_seed": 5,
            "plot_seed": 6,
        },
        "price_process_config": {
            "base_price": 50.0,
            "seasonal_amplitude": 0.25,
        },
        "agent_config": {
            "ppo": {"policy": "MlpPolicy", "n_steps": 8},
            "sac": {"policy": "MlpPolicy"},
            "td3": {"policy": "MlpPolicy"},
        },
        "logging_config": {"run_dir": "runs"},
    }


def test_effective_run_config_keeps_only_used_training_fields() -> None:
    """Effective configs omit benchmark, plotting, and unused agent settings."""
    effective = build_effective_run_config(_base_config(), "ppo", seed_index=3)

    assert "plotting_config" not in effective
    assert effective["evaluation_config"] == {
        "deterministic": True,
        "evaluation_split": "validation",
    }
    assert "lsmc_action_grid" not in effective["evaluation_config"]
    assert effective["seeds"] == {
        "master_seed": 1,
        "dataset_seed": 2,
        "env_seed": 3,
        "agent_seed": 4,
        "eval_seed": 5,
    }
    assert effective["training_config"]["seed_index"] == 3
    assert effective["training_config"]["n_seeds"] == 10
    assert effective["agent_config"] == {
        "algorithm_name": "ppo",
        "hyperparameters": {"policy": "MlpPolicy", "n_steps": 8},
    }


def test_effective_run_config_nests_price_process_parameters() -> None:
    """Effective configs log price parameters under price_process_config."""
    effective = build_effective_run_config(_base_config(), "td3")

    assert effective["price_process_config"] == {
        "source": "config",
        "environment_name": "ou",
        "parameters": {
            "base_price": 50.0,
            "seasonal_amplitude": 0.25,
        },
    }


def test_nested_price_process_config_can_be_loaded_again() -> None:
    """Saved effective configs remain compatible with environment building."""
    effective = build_effective_run_config(_base_config(), "sac")

    price_config = _build_price_process_config(effective)

    assert price_config == {
        "base_price": 50.0,
        "seasonal_amplitude": 0.25,
        "_environment_name": "ou",
    }
