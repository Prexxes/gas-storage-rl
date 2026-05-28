"""Factory for Stable-Baselines3 PPO, SAC, and TD3 agents."""

from __future__ import annotations

from typing import Any


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

        return TD3(policy, env, seed=seed, verbose=verbose, **agent_config)
    raise ValueError(f"Unsupported algorithm_name: {algorithm_name}")
