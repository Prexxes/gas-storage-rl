"""PPO smoke training test."""

import importlib.util

import pytest

from gas_storage_rl.agents.sb3_factory import make_sb3_agent
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.training.config import build_environment, load_config


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("stable_baselines3") is None,
    reason="stable-baselines3 is not installed",
)


def test_train_ppo_smoke() -> None:
    """PPO trains for a tiny number of steps."""
    config = load_config("configs/debug.yaml")
    dataset, params, env_kwargs = build_environment(config)
    env = GasStorageEnv(dataset, "train", params, **env_kwargs)
    model = make_sb3_agent("ppo", env, config["agent_config"]["ppo"], seed=1)
    model.learn(total_timesteps=8)
