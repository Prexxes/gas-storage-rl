"""Factory for Stable-Baselines3 PPO, SAC, and TD3 agents."""

from __future__ import annotations

from typing import Any


def _resolve_activation_fn(name: str) -> Any:
    """Returns a PyTorch activation class for a config-safe activation name."""
    import torch.nn as nn

    activation_functions = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    try:
        return activation_functions[name]
    except KeyError as error:
        raise ValueError(f"Unsupported activation_fn: {name}") from error


def _normalize_policy_kwargs(agent_config: dict[str, Any]) -> None:
    """Converts JSON-safe policy kwargs into SB3 constructor objects."""
    policy_kwargs = agent_config.get("policy_kwargs")
    if not isinstance(policy_kwargs, dict):
        return
    normalized = dict(policy_kwargs)
    activation_fn = normalized.get("activation_fn")
    if isinstance(activation_fn, str):
        normalized["activation_fn"] = _resolve_activation_fn(activation_fn)
    agent_config["policy_kwargs"] = normalized


def _normalize_action_noise(agent_config: dict[str, Any], env: Any) -> None:
    """Converts JSON-safe action-noise config into a TD3 action-noise object."""
    action_noise = agent_config.get("action_noise")
    if action_noise is None or not isinstance(action_noise, dict):
        return
    noise_type = action_noise.get("type", "normal")
    if noise_type != "normal":
        raise ValueError(f"Unsupported action_noise type: {noise_type}")
    import numpy as np
    from stable_baselines3.common.noise import NormalActionNoise

    action_dim = int(np.prod(env.action_space.shape))
    sigma = float(action_noise["sigma"])
    agent_config["action_noise"] = NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=sigma * np.ones(action_dim),
    )


def make_sb3_agent(
    algorithm_name: str,
    env: Any,
    agent_config: dict[str, Any] | None = None,
    seed: int = 0,
) -> Any:
    """Creates an initialized SB3 model.

    Args:
        algorithm_name: One of ``ppo``, ``sac``, or ``td3``.
        env: Gymnasium-compatible environment.
        agent_config: Hyperparameter dictionary passed to the SB3 constructor.
        seed: Agent seed.

    Returns:
        Stable-Baselines3 model instance.
    """
    agent_config = dict(agent_config or {})
    _normalize_policy_kwargs(agent_config)
    policy = agent_config.pop("policy", "MlpPolicy")
    verbose = agent_config.pop("verbose", 0)
    if algorithm_name == "ppo":
        from stable_baselines3 import PPO

        return PPO(policy, env, seed=seed, verbose=verbose, **agent_config)
    if algorithm_name == "sac":
        from stable_baselines3 import SAC

        return SAC(policy, env, seed=seed, verbose=verbose, **agent_config)
    if algorithm_name == "td3":
        from stable_baselines3 import TD3

        _normalize_action_noise(agent_config, env)
        return TD3(policy, env, seed=seed, verbose=verbose, **agent_config)
    raise ValueError(f"Unsupported algorithm_name: {algorithm_name}")
