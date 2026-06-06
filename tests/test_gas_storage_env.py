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
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert info["path_id"] == 0
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0], dtype=np.float32)
    )
    assert obs.shape == (6,)
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
        prices=np.array([[10.0, 10.0]], dtype=np.float32),
        storage_params=StorageParams(
            capacity=5.0,
            withdrawal_rate=0.0,
            initial_inventory=4.0,
        ),
        feasibility_penalty_factor=1.0,
    )
    env.reset()
    _, reward, terminated, _, info = env.step([1.0])
    assert not terminated
    assert info["max_withdrawable_remaining"] == 0.0
    assert info["excess_inventory"] == 1.0
    assert info["feasibility_penalty"] == -10.0
    assert reward == -1.0


def test_feasibility_penalty_for_inventory_that_cannot_be_injected() -> None:
    """Non-terminal reward penalizes inventory below the reachable target."""
    env = make_env(
        prices=np.array([[10.0, 10.0]], dtype=np.float32),
        storage_params=StorageParams(
            capacity=5.0,
            injection_rate=0.0,
            initial_inventory=4.0,
        ),
        feasibility_penalty_factor=1.0,
    )
    env.reset(options={"initial_inventory": 4.0})

    _, reward, terminated, _, info = env.step([-1.0])

    assert not terminated
    assert info["max_injectable_remaining"] == 0.0
    assert info["inventory_shortfall"] == 1.0
    assert info["feasibility_penalty"] == -10.0
    assert reward == -1.0


def test_normalized_observation_values() -> None:
    """Initial observation is normalized as specified."""
    env = make_env()
    obs, _ = env.reset()
    assert np.allclose(
        obs,
        np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def test_observation_uses_path_calendar_date() -> None:
    """Calendar features reflect a historical path's actual start date."""
    prices = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    dataset = PathDataset(
        {"train": prices},
        {"train": 1},
        {"train": [{"start_date": "2025-06-06", "end_date": "2025-06-08"}]},
    )
    env = GasStorageEnv(
        dataset,
        "train",
        StorageParams(capacity=2.0),
        price_scale=10.0,
        fixed_path_id=0,
    )

    obs, _ = env.reset()

    angle = 2.0 * np.pi * (157 - 1) / 365
    assert np.allclose(obs[2:4], [np.sin(angle), np.cos(angle)])
    assert obs[4] == 1.0


def test_training_reset_samples_contiguous_windows_reproducibly() -> None:
    """Training resets sample starts while preserving contiguous prices."""
    raw_path = np.arange(1.0, 8.0, dtype=np.float32).reshape(1, -1)
    dataset = PathDataset(
        {"train": raw_path},
        {"train": 1},
        episode_length_override=4,
        start_indices_by_split={"train": np.array([2])},
        base_dates_by_split={"train": "2001-01-01"},
    )
    env = GasStorageEnv(
        dataset,
        "train",
        StorageParams(capacity=2.0),
        price_scale=1.0,
    )

    obs, info = env.reset(seed=7)
    expected_start = int(np.random.default_rng(7).integers(0, 4))

    assert info["start_index"] == expected_start
    assert obs[1] == raw_path[0, expected_start]
    next_obs, _, _, _, _ = env.step([0.0])
    assert next_obs[1] == raw_path[0, expected_start + 1]


def test_explicit_path_id_uses_fixed_start_index() -> None:
    """Evaluation resets use the stored deterministic start per path."""
    raw_path = np.arange(1.0, 8.0, dtype=np.float32).reshape(1, -1)
    dataset = PathDataset(
        {"train": raw_path, "validation": raw_path},
        {"train": 1, "validation": 2},
        episode_length_override=4,
        start_indices_by_split={
            "train": np.array([0]),
            "validation": np.array([3]),
        },
        base_dates_by_split={
            "train": "2001-01-01",
            "validation": "2001-01-01",
        },
    )
    env = GasStorageEnv(
        dataset,
        "validation",
        StorageParams(capacity=2.0),
        price_scale=1.0,
    )

    obs, info = env.reset(options={"path_id": 0})

    assert info["start_index"] == 3
    assert obs[1] == 4.0


def test_training_reset_samples_initial_inventory_and_sets_target() -> None:
    """Training samples one inventory and uses it as the episode target."""
    env = make_env()
    env.fixed_path_id = None
    env.initial_inventory_mean_fraction = 0.30
    env.initial_inventory_std_fraction = 0.05

    obs, info = env.reset(seed=11)

    assert 0.0 <= info["initial_inventory"] <= 2.0
    assert info["initial_inventory"] != 0.0
    assert info["target_terminal_inventory"] == info["initial_inventory"]
    expected_fraction = info["initial_inventory"] / 2.0
    assert np.isclose(obs[0], expected_fraction)
    assert np.isclose(obs[5], expected_fraction)


def test_fixed_inventory_is_used_for_evaluation_path() -> None:
    """Evaluation uses the inventory stored for the selected path."""
    prices = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    dataset = PathDataset(
        {"train": prices, "validation": prices},
        {"train": 1, "validation": 2},
        initial_inventories_by_split={
            "train": np.array([0.5]),
            "validation": np.array([0.8]),
        },
    )
    env = GasStorageEnv(
        dataset,
        "validation",
        StorageParams(capacity=2.0),
        price_scale=10.0,
    )

    obs, info = env.reset(options={"path_id": 0})

    assert info["initial_inventory"] == 0.8
    assert info["target_terminal_inventory"] == 0.8
    assert np.isclose(obs[0], 0.4)
    assert np.isclose(obs[5], 0.4)
