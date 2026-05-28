"""Tests for the Gymnasium gas storage environment."""

import numpy as np

from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.envs.storage_dynamics import StorageParams


def make_env(prices: np.ndarray | None = None) -> GasStorageEnv:
    """Creates a tiny deterministic environment."""
    price_paths = np.asarray(prices if prices is not None else [[10.0, 20.0, 30.0]], dtype=np.float32)
    dataset = PathDataset(
        {"train": price_paths, "validation": price_paths, "test": price_paths},
        {"train": 1, "validation": 2, "test": 3},
    )
    return GasStorageEnv(
        dataset,
        "train",
        StorageParams(capacity=2.0),
        price_scale=10.0,
        reward_scale=10.0,
        penalty_factor=0.5,
        fixed_path_id=0,
    )


def test_gymnasium_reset_step_api_shape_and_dtype() -> None:
    """Reset and step follow the Gymnasium API."""
    env = make_env()
    obs, info = env.reset(seed=1)
    assert obs.shape == (3,)
    assert obs.dtype == np.float32
    assert info["path_id"] == 0
    obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert obs.shape == (3,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert info["split"] == "train"


def test_buy_sell_hold_reward_signs() -> None:
    """Buying is negative, selling is positive, holding is zero."""
    env = make_env()
    env.reset()
    _, buy_reward, _, _, buy_info = env.step([1.0])
    _, sell_reward, _, _, sell_info = env.step([-1.0])
    _, hold_reward, _, _, hold_info = env.step([0.0])
    assert buy_info["raw_cashflow"] < 0.0
    assert sell_info["raw_cashflow"] > 0.0
    assert hold_info["raw_cashflow"] == 0.0
    assert buy_reward < 0.0
    assert sell_reward > 0.0
    assert hold_reward == hold_info["scaled_reward"]


def test_terminal_penalty_and_scaled_reward() -> None:
    """Final reward includes terminal penalty and scaling."""
    env = make_env()
    env.reset()
    env.step([1.0])
    env.step([0.0])
    _, reward, terminated, _, info = env.step([0.0])
    assert terminated
    assert info["terminal_penalty"] < 0.0
    assert np.isclose(reward, info["raw_reward"] / 10.0)


def test_normalized_observation_values() -> None:
    """Initial observation is normalized as specified."""
    env = make_env()
    obs, _ = env.reset()
    assert np.allclose(obs, np.array([0.0, 1.0, 0.0], dtype=np.float32))
