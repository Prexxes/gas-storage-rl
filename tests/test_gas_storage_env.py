"""Tests for the Gymnasium gas storage environment."""

import numpy as np
from gymnasium.utils.env_checker import check_env as gymnasium_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.envs.storage_dynamics import StorageParams


def make_env(
    prices: np.ndarray | None = None,
    storage_params: StorageParams | None = None,
    **env_kwargs,
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
        fixed_path_id=0,
        **env_kwargs,
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


def test_environment_passes_gymnasium_and_sb3_checks() -> None:
    """Environment satisfies the Gymnasium and SB3 environment contracts."""
    env = make_env()

    gymnasium_check_env(env)
    sb3_check_env(env, warn=True)


def test_cashflow_reward_is_used_for_training() -> None:
    """Training reward follows immediate cashflow."""
    env = make_env()
    env.reset()
    _, buy_reward, _, _, buy_info = env.step([1.0])
    _, sell_reward, _, _, sell_info = env.step([-1.0])
    assert buy_info["raw_cashflow"] < 0.0
    assert sell_info["raw_cashflow"] > 0.0
    assert buy_info["raw_reward"] == -10.0
    assert sell_info["raw_reward"] == 20.0
    assert buy_reward == buy_info["scaled_reward"]
    assert sell_reward == sell_info["scaled_reward"]


def test_final_action_is_clipped_to_avoid_terminal_penalty() -> None:
    """Final action is clipped to the target when physically reachable."""
    env = make_env()
    env.reset()
    env.step([1.0])
    env.step([0.0])
    _, reward, terminated, _, info = env.step([0.0])
    assert terminated
    assert info["executed_action"] == -1.0
    assert info["terminal_feasibility_clipped"] is True
    assert info["terminal_penalty"] == 0.0
    assert np.isclose(info["raw_reward"], 30.0)
    assert np.isclose(reward, info["raw_reward"] / 10.0)


def test_terminal_penalty_remains_when_target_is_physically_unreachable() -> None:
    """Terminal penalty still applies when rate limits prevent reaching target."""
    env = make_env(
        prices=np.array([[10.0]], dtype=np.float32),
        storage_params=StorageParams(
            capacity=5.0,
            withdrawal_rate=0.0,
            initial_inventory=1.0,
        ),
    )
    env.reset()
    env.target_terminal_inventory = 0.0

    _, reward, terminated, _, info = env.step([0.0])

    assert terminated
    assert info["executed_action"] == 0.0
    assert info["terminal_penalty"] < 0.0
    assert np.isclose(info["raw_reward"], info["terminal_penalty"])
    assert np.isclose(reward, info["raw_reward"] / 10.0)


def test_terminal_feasibility_prevents_unwithdrawable_purchase() -> None:
    """Terminal-feasibility clipping prevents excess inventory from a purchase."""
    env = make_env(
        prices=np.array([[10.0, 10.0]], dtype=np.float32),
        storage_params=StorageParams(
            capacity=5.0,
            withdrawal_rate=0.0,
            initial_inventory=4.0,
        ),
    )
    env.reset()
    _, reward, terminated, _, info = env.step([1.0])
    assert not terminated
    assert info["rate_capacity_clipped_action"] == 1.0
    assert info["executed_action"] == 0.0
    assert info["terminal_feasibility_clipped"] is True
    assert reward == 0.0


def test_terminal_feasibility_prevents_uninjectable_sale() -> None:
    """Terminal-feasibility clipping prevents shortfall from a sale."""
    env = make_env(
        prices=np.array([[10.0, 10.0]], dtype=np.float32),
        storage_params=StorageParams(
            capacity=5.0,
            injection_rate=0.0,
            initial_inventory=4.0,
        ),
    )
    env.reset(options={"initial_inventory": 4.0})

    _, reward, terminated, _, info = env.step([-1.0])

    assert not terminated
    assert info["rate_capacity_clipped_action"] == -1.0
    assert info["executed_action"] == 0.0
    assert info["terminal_feasibility_clipped"] is True
    assert reward == 0.0


def test_terminal_feasibility_clips_final_action_to_target() -> None:
    """On the final step the executed action is clipped toward the target."""
    env = make_env(
        prices=np.array([[10.0]], dtype=np.float32),
        storage_params=StorageParams(capacity=5.0, initial_inventory=1.0),
    )
    env.reset()
    env.target_terminal_inventory = 0.0

    _, _, terminated, _, info = env.step([0.0])

    assert terminated
    assert info["rate_capacity_clipped_action"] == 0.0
    assert info["executed_action"] == -1.0
    assert info["terminal_feasibility_clipped"] is True
    assert info["storage_level"] == 0.0
    assert info["terminal_penalty"] == 0.0


def test_v2_penalizes_only_terminal_clip_distance() -> None:
    """V2 shapes training reward with terminal clip penalties."""
    env = make_env(
        prices=np.array([[10.0]], dtype=np.float32),
        clipping_variant="v2",
        clip_penalty_factor=1.0,
    )
    env.reset(options={"initial_inventory": 0.0})

    _, reward, terminated, _, info = env.step([1.0])

    assert terminated
    assert info["clipping_variant"] == "v2"
    assert info["rate_capacity_clipped_action"] == 1.0
    assert info["executed_action"] == 0.0
    assert info["terminal_clip_distance"] == 1.0
    assert info["raw_reward"] == 0.0
    assert info["clip_penalty"] == -10.0
    assert info["shaped_raw_reward"] == -10.0
    assert reward == -1.0


def test_normalized_observation_values() -> None:
    """Initial observation is normalized as specified."""
    env = make_env()
    obs, _ = env.reset()
    assert np.allclose(
        obs,
        np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def test_observation_features_can_be_masked() -> None:
    """Configured feature ablations keep the six-dimensional observation shape."""
    prices = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    dataset = PathDataset(
        {"train": prices},
        {"train": 1},
        {"train": [{"start_date": "2025-06-06", "end_date": "2025-06-08"}]},
        initial_inventories_by_split={"train": np.array([1.0])},
    )
    env = GasStorageEnv(
        dataset,
        "train",
        StorageParams(capacity=2.0),
        price_scale=10.0,
        observation_features={
            "inventory": False,
            "price": False,
            "calendar": False,
            "remaining_time": False,
            "target_inventory": False,
        },
        fixed_path_id=0,
    )

    obs, _ = env.reset()

    assert obs.shape == (6,)
    assert np.allclose(
        obs,
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )


def test_unknown_observation_feature_is_rejected() -> None:
    """Observation masks fail fast for misspelled feature names."""
    dataset = PathDataset(
        {"train": np.array([[10.0]], dtype=np.float32)},
        {"train": 1},
    )

    try:
        GasStorageEnv(
            dataset,
            "train",
            StorageParams(capacity=2.0),
            observation_features={"calender": False},
        )
    except ValueError as exc:
        assert "calender" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown observation feature")


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


def test_observation_space_accepts_negative_prices() -> None:
    """Additive synthetic prices may be negative in observations."""
    prices = np.array([[-1.0, 0.0]], dtype=np.float32)
    dataset = PathDataset({"train": prices}, {"train": 1})
    env = GasStorageEnv(
        dataset,
        "train",
        StorageParams(capacity=2.0),
        price_scale=2.0,
    )

    observation, _ = env.reset()

    assert observation[1] == -0.5
    assert env.observation_space.contains(observation)


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
