"""Optuna search spaces for SB3 hyperparameter tuning."""

from __future__ import annotations

from typing import Any

FIXED_GAMMA = 1.0
REWARD_SCALE_MULTIPLIERS = [1.0, 2.0, 4.0]
TRAIN_UPDATE_COMBOS = {
    "tf1_gs1": {"train_freq": 1, "gradient_steps": 1},
    "tf4_gs1": {"train_freq": 4, "gradient_steps": 1},
    "tf4_gs2": {"train_freq": 4, "gradient_steps": 2},
    "tf8_gs1": {"train_freq": 8, "gradient_steps": 1},
    "tf8_gs2": {"train_freq": 8, "gradient_steps": 2},
}
BUFFER_SIZES = [100_000, 250_000, 1_000_000]

REFERENCE_TRIAL_PARAMS = {
    "ppo": {
        "reward_scale_multiplier": 1.0,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "clip_range_vf_mode": "none",
        "target_kl_mode": "none",
        "net_arch": "ppo_default",
        "activation_fn": "tanh",
        "ortho_init": True,
    },
    "sac": {
        "reward_scale_multiplier": 1.0,
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "batch_size": 256,
        "tau": 0.005,
        "train_update_combo": "tf1_gs1",
        "ent_coef": "auto",
        "net_arch": "sac_default",
        "activation_fn": "relu",
    },
    "td3": {
        "reward_scale_multiplier": 1.0,
        "learning_rate": 1e-3,
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "batch_size": 256,
        "tau": 0.005,
        "train_update_combo": "tf1_gs1",
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "action_noise_mode": "none",
        "net_arch": "td3_default",
        "activation_fn": "relu",
    },
}


def suggest_hyperparameters(trial: Any, algorithm: str) -> dict[str, Any]:
    """Suggests JSON-serializable SB3 hyperparameters for one algorithm.
    
    Args:
        trial: Optuna trial-like object.
        algorithm: One of ``ppo``, ``sac``, or ``td3``.
    
    Returns:
        Hyperparameters suitable for ``agent_config[algorithm]``.
    
    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    if algorithm == "ppo":
        return suggest_ppo_hyperparameters(trial)
    if algorithm == "sac":
        return suggest_sac_hyperparameters(trial)
    if algorithm == "td3":
        return suggest_td3_hyperparameters(trial)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def suggest_reward_scale_multiplier(trial: Any) -> float:
    """Suggests the shared reward-scale multiplier for one HPO trial.
    
    Args:
        trial: Optuna trial object or fixed-trial adapter.
    
    Returns:
        Computed result.

    """
    return float(
        trial.suggest_categorical(
            "reward_scale_multiplier",
            REWARD_SCALE_MULTIPLIERS,
        )
    )


def reference_trial_params(algorithm: str) -> dict[str, Any]:
    """Returns SB3-default reference-trial parameters for one algorithm.

    Args:
        algorithm: One of ``ppo``, ``sac``, or ``td3``.

    Returns:
        Optuna parameter dictionary for an enqueued reference trial.

    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    try:
        return dict(REFERENCE_TRIAL_PARAMS[algorithm])
    except KeyError as error:
        raise ValueError(f"Unsupported algorithm: {algorithm}") from error


def _suggest_train_update_combo(trial: Any) -> dict[str, int]:
    """Suggests a bounded train/update cadence for off-policy algorithms.

    Args:
        trial: Optuna trial object or fixed-trial adapter.

    Returns:
        Train frequency and gradient step settings.

    """
    combo_name = trial.suggest_categorical(
        "train_update_combo",
        list(TRAIN_UPDATE_COMBOS),
    )
    return dict(TRAIN_UPDATE_COMBOS[combo_name])


def suggest_ppo_hyperparameters(trial: Any) -> dict[str, Any]:
    """Suggests PPO-specific hyperparameters.
    
    Args:
        trial: Optuna trial object or fixed-trial adapter.
    
    Returns:
        Computed result.

    """
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    clip_range_vf_mode = trial.suggest_categorical(
        "clip_range_vf_mode",
        ["none", "tuned"],
    )
    target_kl_mode = trial.suggest_categorical("target_kl_mode", ["none", "tuned"])
    hyperparameters: dict[str, Any] = {
        "policy": "MlpPolicy",
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 15]),
        "gamma": FIXED_GAMMA,
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 1.00),
        "clip_range": trial.suggest_float("clip_range", 0.10, 0.30),
        "normalize_advantage": True,
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 1e-4, 1e-3, 1e-2]),
        "vf_coef": trial.suggest_float("vf_coef", 0.25, 0.75),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        "policy_kwargs": _suggest_policy_kwargs(
            trial,
            algorithm="ppo",
            include_ortho_init=True,
        ),
    }
    if clip_range_vf_mode == "tuned":
        hyperparameters["clip_range_vf"] = trial.suggest_float(
            "clip_range_vf",
            0.10,
            0.30,
        )
    else:
        hyperparameters["clip_range_vf"] = None
    if target_kl_mode == "tuned":
        hyperparameters["target_kl"] = trial.suggest_float(
            "target_kl",
            0.01,
            0.20,
        )
    else:
        hyperparameters["target_kl"] = None
    return hyperparameters


def suggest_sac_hyperparameters(trial: Any) -> dict[str, Any]:
    """Suggests SAC-specific hyperparameters.
    
    Args:
        trial: Optuna trial object or fixed-trial adapter.
    
    Returns:
        Computed result.

    """
    train_update_combo = _suggest_train_update_combo(trial)
    hyperparameters: dict[str, Any] = {
        "policy": "MlpPolicy",
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical(
            "buffer_size",
            BUFFER_SIZES,
        ),
        "learning_starts": trial.suggest_categorical(
            "learning_starts",
            [100, 1_000, 5_000],
        ),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "tau": trial.suggest_categorical("tau", [0.005, 0.01, 0.02]),
        "gamma": FIXED_GAMMA,
        "train_freq": train_update_combo["train_freq"],
        "gradient_steps": train_update_combo["gradient_steps"],
        "ent_coef": trial.suggest_categorical(
            "ent_coef",
            ["auto", "auto_0.5", "auto_0.1"],
        ),
        "target_entropy": "auto",
        "use_sde": False,
        "sde_sample_freq": -1,
        "policy_kwargs": _suggest_policy_kwargs(trial, algorithm="sac"),
    }
    return hyperparameters


def suggest_td3_hyperparameters(trial: Any) -> dict[str, Any]:
    """Suggests TD3-specific hyperparameters.
    
    Args:
        trial: Optuna trial object or fixed-trial adapter.
    
    Returns:
        Computed result.

    """
    action_noise_mode = trial.suggest_categorical(
        "action_noise_mode",
        ["none", "normal"],
    )
    train_update_combo = _suggest_train_update_combo(trial)
    hyperparameters: dict[str, Any] = {
        "policy": "MlpPolicy",
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True),
        "buffer_size": trial.suggest_categorical(
            "buffer_size",
            BUFFER_SIZES,
        ),
        "learning_starts": trial.suggest_categorical(
            "learning_starts",
            [100, 1_000, 5_000],
        ),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "tau": trial.suggest_categorical("tau", [0.005, 0.01, 0.02]),
        "gamma": FIXED_GAMMA,
        "train_freq": train_update_combo["train_freq"],
        "gradient_steps": train_update_combo["gradient_steps"],
        "policy_delay": 2,
        "target_policy_noise": trial.suggest_float(
            "target_policy_noise",
            0.10,
            0.30,
        ),
        "target_noise_clip": trial.suggest_float("target_noise_clip", 0.30, 0.70),
        "policy_kwargs": _suggest_policy_kwargs(trial, algorithm="td3"),
    }
    if action_noise_mode == "normal":
        hyperparameters["action_noise"] = {
            "type": "normal",
            "sigma": trial.suggest_categorical("action_noise_sigma", [0.1, 0.2]),
        }
    return hyperparameters


def search_space_description() -> dict[str, Any]:
    """Returns a human-readable summary of the configured search spaces.
    
    Returns:
        Computed result.

    """
    return {
        "method": "Optuna TPE",
        "direction": "maximize",
        "shared_tuned_parameters": {
            "reward_scale_multiplier": (
                "categorical [1.0, 2.0, 4.0]; effective "
                "reward_scale = base reward_scale * multiplier"
            ),
        },
        "fixed_design_parameters": [
            "price_process_config",
            "dataset_config",
            "reward_function",
            "storage_restrictions",
            "capacity",
            "terminal_target",
            "benchmarks",
            "evaluation_metrics",
            "total_timesteps",
            "seed_counts",
            "gamma = 1.0",
        ],
        "ppo": {
            "learning_rate": "float log [1e-4, 1e-3]",
            "n_steps": "categorical [1024, 2048, 4096]",
            "batch_size": "categorical [64, 128, 256], <= n_steps",
            "n_epochs": "categorical [5, 10, 15]",
            "gamma": "fixed 1.0",
            "gae_lambda": "float linear [0.90, 1.00]",
            "clip_range": "float linear [0.10, 0.30]",
            "clip_range_vf": "None or float linear [0.10, 0.30]",
            "normalize_advantage": "fixed true",
            "ent_coef": "categorical [0.0, 0.0001, 0.001, 0.01]",
            "vf_coef": "float linear [0.25, 0.75]",
            "max_grad_norm": "float linear [0.3, 1.0]",
            "target_kl": "None or float linear [0.01, 0.20]",
            "policy_kwargs": "net_arch ['ppo_default', 'small', 'medium'], activation_fn, ortho_init",
        },
        "sac": {
            "learning_rate": "float log [1e-4, 1e-3]",
            "buffer_size": "categorical [100000, 250000, 500000, 1000000]",
            "learning_starts": "categorical [100, 1000, 5000]",
            "batch_size": "categorical [128, 256]",
            "tau": "categorical [0.005, 0.01, 0.02]",
            "gamma": "fixed 1.0",
            "train_update_combo": (
                "categorical ['tf1_gs1', 'tf1_gs2', 'tf4_gs1', 'tf4_gs2', "
                "'tf8_gs1', 'tf8_gs2']"
            ),
            "ent_coef": "categorical ['auto', 'auto_0.5', 'auto_0.1']",
            "target_entropy": "fixed 'auto'",
            "use_sde": "fixed false",
            "sde_sample_freq": "fixed -1",
            "policy_kwargs": "net_arch ['sac_default', 'small'], activation_fn",
        },
        "td3": {
            "learning_rate": "float log [3e-4, 2e-3]",
            "buffer_size": "categorical [100000, 250000, 500000, 1000000]",
            "learning_starts": "categorical [100, 1000, 5000]",
            "batch_size": "categorical [128, 256]",
            "tau": "categorical [0.005, 0.01, 0.02]",
            "gamma": "fixed 1.0",
            "train_update_combo": (
                "categorical ['tf1_gs1', 'tf1_gs2', 'tf4_gs1', 'tf4_gs2', "
                "'tf8_gs1', 'tf8_gs2']"
            ),
            "policy_delay": "fixed 2",
            "target_policy_noise": "float linear [0.10, 0.30]",
            "target_noise_clip": "float linear [0.30, 0.70]",
            "action_noise": "None or normal sigma categorical [0.1, 0.2]",
            "policy_kwargs": "net_arch ['td3_default', 'small', 'medium'], activation_fn",
        },
    }


def _suggest_policy_kwargs(
    trial: Any,
    *,
    algorithm: str,
    include_ortho_init: bool = False,
) -> dict[str, Any]:
    """Suggests JSON-safe policy keyword arguments.
    
    Args:
        trial: Optuna trial object or fixed-trial adapter.
        include_ortho_init: Include ortho init value.
    
    Returns:
        Suggest policy kwargs result.

    """
    choices_by_algorithm = {
        "ppo": ["ppo_default", "small", "medium"],
        "sac": ["sac_default", "small"],
        "td3": ["td3_default", "small", "medium"],
    }
    net_arch_by_name = {
        "ppo_default": [64, 64],
        "sac_default": [256, 256],
        "td3_default": [400, 300],
        "small": [128, 128],
        "medium": [256, 256],
    }
    try:
        net_arch_choices = choices_by_algorithm[algorithm]
    except KeyError as error:
        raise ValueError(f"Unsupported algorithm: {algorithm}") from error
    net_arch_name = trial.suggest_categorical("net_arch", net_arch_choices)
    policy_kwargs: dict[str, Any] = {
        "net_arch": net_arch_by_name[net_arch_name],
        "activation_fn": trial.suggest_categorical(
            "activation_fn",
            ["tanh", "relu"],
        ),
    }
    if include_ortho_init:
        policy_kwargs["ortho_init"] = trial.suggest_categorical(
            "ortho_init",
            [True, False],
        )
    return policy_kwargs
