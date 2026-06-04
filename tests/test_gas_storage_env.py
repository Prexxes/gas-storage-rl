"""Tests for the Gymnasium gas storage environment."""

import numpy as np

from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.envs.storage_dynamics import StorageParams


def make_env(
    prices: np.ndarray | None = None,
    storage_params: StorageParams | None = None,
    feasibility_penalty_factor: float = 0.5,
) -> GasStorageEnv:
    """Creates a tiny deterministic environment."""
    price_paths = np.asarray(
        prices if prices is not None else [[10.0, 20.0, 30.0]],
        dtype=np.float32,
    )
    dataset = PathDataset(
        {"train": price_paths, "validation": price_paths, "test": price_paths},
        {"train": 1, "validation": 2, "test": 3},
    )
    return GasStorageEnv(
        dataset,
        "train",
        storage_params or StorageParams(capacity=2.0),
        price_scale=10.0,
        reward_scale=10.0,
        penalty_factor=0.5,
        feasibility_penalty_factor=feasibility_penalty_factor,
        fixed_path_id=0,
    )


def test_gymnasium_reset_step_api_shape_and_dtype() -> None:
    """Reset and step follow the Gymnasium API."""
    env = make_env()
    obs, info = env.reset(seed=1)
    assert obs.shape == (3,)
    assert obs.dtype == np.float32
    assert info["path_id"] == 0
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0], dtype=np.float32)
    )
    assert obs.shape == (3,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert info["split"] == "train"


def test_mark_to_market_reward_is_used_for_training() -> None:
    """Training reward follows marked inventory value, not immediate cashflow."""
    env = make_env()
    env.reset()
    _, buy_reward, _, _, buy_info = env.step([1.0])
    _, sell_reward, _, _, sell_info = env.step([-1.0])
    assert buy_info["raw_cashflow"] < 0.0
    assert sell_info["raw_cashflow"] > 0.0
    assert buy_info["mark_to_market_reward"] == 10.0
    assert buy_info["economic_reward_raw"] == -10.0
    assert buy_info["shaped_reward_raw"] == 10.0
    assert buy_reward == buy_info["scaled_reward"]
    assert sell_reward == sell_info["scaled_reward"]


def test_terminal_penalty_and_scaled_reward() -> None:
    """Final shaped reward includes cashflow, terminal penalty, and inventory value."""
    env = make_env()
    env.reset()
    env.step([1.0])
    env.step([0.0])
    _, reward, terminated, _, info = env.step([0.0])
    assert terminated
    assert info["terminal_penalty"] < 0.0
    assert np.isclose(info["economic_reward_raw"], info["terminal_penalty"])
    assert np.isclose(info["shaped_reward_raw"], info["terminal_penalty"] - 30.0)
    assert np.isclose(reward, info["shaped_reward_raw"] / 10.0)


def test_feasibility_penalty_for_inventory_that_cannot_be_withdrawn() -> None:
    """Non-terminal shaped reward penalizes inventory above withdrawable volume."""
    env = make_env(
        prices=np.array([[10.0, 10.0, 10.0]], dtype=np.float32),
        storage_params=StorageParams(capacity=5.0, initial_inventory=4.0),
        feasibility_penalty_factor=1.0,
    )
    env.reset()
    _, reward, terminated, _, info = env.step([0.0])
    assert not terminated
    assert info["max_withdrawable_remaining"] == 2.0
    assert info["excess_inventory"] == 2.0
    assert info["feasibility_penalty"] == -20.0
    assert reward == -2.0


def test_normalized_observation_values() -> None:
    """Initial observation is normalized as specified."""
    env = make_env()
    obs, _ = env.reset()
    assert np.allclose(obs, np.array([0.0, 1.0, 0.0], dtype=np.float32))
