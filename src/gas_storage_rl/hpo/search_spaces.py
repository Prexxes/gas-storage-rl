"""Optuna search spaces for SB3 hyperparameter tuning."""

from __future__ import annotations

from typing import Any

REWARD_SCALE_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]
TRAIN_UPDATE_COMBOS = {
    "tf1_gs1": {"train_freq": 1, "gradient_steps": 1},
    "tf4_gs1": {"train_freq": 4, "gradient_steps": 1},
    "tf4_gs2": {"train_freq": 4, "gradient_steps": 2},
    "tf8_gs1": {"train_freq": 8, "gradient_steps": 1},
    "tf8_gs2": {"train_freq": 8, "gradient_steps": 2},
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
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    clip_range_vf_mode = trial.suggest_categorical(
        "clip_range_vf_mode",
        ["none", "tuned"],
    )
    target_kl_mode = trial.suggest_categorical("target_kl_mode", ["none", "tuned"])
    hyperparameters: dict[str, Any] = {
        "policy": "MlpPolicy",
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.80, 1.00),
        "clip_range": trial.suggest_float("clip_range", 0.05, 0.30),
        "normalize_advantage": trial.suggest_categorical(
            "normalize_advantage",
            [True, False],
        ),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
        "policy_kwargs": _suggest_policy_kwargs(
            trial,
            include_ortho_init=True,
            include_log_std_init=True,
        ),
    }
    if clip_range_vf_mode == "tuned":
        hyperparameters["clip_range_vf"] = trial.suggest_float(
            "clip_range_vf",
            0.05,
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
    ent_coef_mode = trial.suggest_categorical("ent_coef_mode", ["auto", "fixed"])
    train_update_combo = _suggest_train_update_combo(trial)
    hyperparameters: dict[str, Any] = {
        "policy": "MlpPolicy",
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "buffer_size": trial.suggest_categorical(
            "buffer_size",
            [100_000, 250_000, 500_000, 1_000_000],
        ),
        "learning_starts": trial.suggest_categorical(
            "learning_starts",
            [1_000, 5_000, 10_000],
        ),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "tau": trial.suggest_float("tau", 0.005, 0.05, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        "train_freq": train_update_combo["train_freq"],
        "gradient_steps": train_update_combo["gradient_steps"],
        "target_entropy": trial.suggest_categorical(
            "target_entropy",
            ["auto", -0.5, -1.0, -2.0],
        ),
        "use_sde": trial.suggest_categorical("use_sde", [False, True]),
        "sde_sample_freq": trial.suggest_categorical(
            "sde_sample_freq",
            [-1, 4, 16, 64],
        ),
        "policy_kwargs": _suggest_policy_kwargs(trial),
    }
    if ent_coef_mode == "auto":
        initial_ent_coef = trial.suggest_float(
            "initial_ent_coef",
            1e-3,
            1.0,
            log=True,
        )
        hyperparameters["ent_coef"] = f"auto_{initial_ent_coef:.8g}"
    else:
        hyperparameters["ent_coef"] = trial.suggest_float(
            "ent_coef",
            1e-4,
            1e-1,
            log=True,
        )
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
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "buffer_size": trial.suggest_categorical(
            "buffer_size",
            [100_000, 250_000, 500_000, 1_000_000],
        ),
        "learning_starts": trial.suggest_categorical(
            "learning_starts",
            [1_000, 5_000, 10_000],
        ),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "tau": trial.suggest_float("tau", 0.005, 0.05, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        "train_freq": train_update_combo["train_freq"],
        "gradient_steps": train_update_combo["gradient_steps"],
        "policy_delay": trial.suggest_categorical("policy_delay", [2, 3]),
        "target_policy_noise": trial.suggest_float(
            "target_policy_noise",
            0.05,
            0.50,
        ),
        "target_noise_clip": trial.suggest_float("target_noise_clip", 0.10, 1.00),
        "policy_kwargs": _suggest_policy_kwargs(trial),
    }
    if action_noise_mode == "normal":
        hyperparameters["action_noise"] = {
            "type": "normal",
            "sigma": trial.suggest_float("action_noise_sigma", 0.05, 0.50),
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
                "categorical [0.25, 0.5, 1.0, 2.0, 4.0]; effective "
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
        ],
        "ppo": {
            "learning_rate": "float log [1e-5, 3e-3]",
            "n_steps": "categorical [256, 512, 1024, 2048]",
            "batch_size": "categorical [64, 128, 256], <= n_steps",
            "n_epochs": "categorical [5, 10, 20]",
            "gamma": "float linear [0.95, 0.9999]",
            "gae_lambda": "float linear [0.80, 1.00]",
            "clip_range": "float linear [0.05, 0.30]",
            "clip_range_vf": "None or float linear [0.05, 0.30]",
            "normalize_advantage": "categorical [true, false]",
            "ent_coef": "float log [1e-8, 1e-2]",
            "vf_coef": "float linear [0.1, 1.0]",
            "max_grad_norm": "float linear [0.3, 5.0]",
            "target_kl": "None or float linear [0.01, 0.20]",
            "policy_kwargs": "net_arch, activation_fn, ortho_init, log_std_init",
        },
        "sac": {
            "learning_rate": "float log [1e-5, 3e-3]",
            "buffer_size": "categorical [100000, 250000, 500000, 1000000]",
            "learning_starts": "categorical [1000, 5000, 10000]",
            "batch_size": "categorical [128, 256]",
            "tau": "float log [0.005, 0.05]",
            "gamma": "float linear [0.95, 0.9999]",
            "train_update_combo": (
                "categorical ['tf1_gs1', 'tf4_gs1', 'tf4_gs2', "
                "'tf8_gs1', 'tf8_gs2']"
            ),
            "ent_coef": "auto with tuned initial value, or fixed log [1e-4, 1e-1]",
            "target_entropy": "categorical ['auto', -0.5, -1.0, -2.0]",
            "use_sde": "categorical [false, true]",
            "sde_sample_freq": "categorical [-1, 4, 16, 64]",
            "policy_kwargs": "net_arch, activation_fn",
        },
        "td3": {
            "learning_rate": "float log [1e-5, 3e-3]",
            "buffer_size": "categorical [100000, 250000, 500000, 1000000]",
            "learning_starts": "categorical [1000, 5000, 10000]",
            "batch_size": "categorical [128, 256]",
            "tau": "float log [0.005, 0.05]",
            "gamma": "float linear [0.95, 0.9999]",
            "train_update_combo": (
                "categorical ['tf1_gs1', 'tf4_gs1', 'tf4_gs2', "
                "'tf8_gs1', 'tf8_gs2']"
            ),
            "policy_delay": "categorical [2, 3]",
            "target_policy_noise": "float linear [0.05, 0.50]",
            "target_noise_clip": "float linear [0.10, 1.00]",
            "action_noise": "None or normal sigma linear [0.05, 0.50]",
            "policy_kwargs": "net_arch, activation_fn",
        },
    }


def _suggest_policy_kwargs(
    trial: Any,
    *,
    include_ortho_init: bool = False,
    include_log_std_init: bool = False,
) -> dict[str, Any]:
    """Suggests JSON-safe policy keyword arguments.
    
    Args:
        trial: Optuna trial object or fixed-trial adapter.
        include_ortho_init: Include ortho init value.
        include_log_std_init: Include log std init value.
    
    Returns:
        Suggest policy kwargs result.

    """
    net_arch_name = trial.suggest_categorical(
        "net_arch",
        ["small", "medium", "large"],
    )
    net_arch_by_name = {
        "small": [128, 128],
        "medium": [256, 256],
        "large": [512, 512],
    }
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
    if include_log_std_init:
        policy_kwargs["log_std_init"] = trial.suggest_float(
            "log_std_init",
            -4.0,
            1.0,
        )
    return policy_kwargs
