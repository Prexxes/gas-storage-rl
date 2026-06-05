"""Tests for behavior-cloning pretraining."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from gas_storage_rl.agents.sb3_factory import make_sb3_agent
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.pretraining.behavior_cloning import (
    load_trajectory_samples,
    run_pretraining,
)
from gas_storage_rl.training.config import build_environment, load_config
from gas_storage_rl.training.run_experiment import load_pretrained_policy_state


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("stable_baselines3") is None,
    reason="stable-baselines3 is not installed",
)


def test_trajectory_samples_include_calendar_and_remaining_time(tmp_path) -> None:
    """Trajectory observations use the five-feature time representation."""
    trajectory_path = tmp_path / "trajectory.jsonl"
    trajectory = {
        "start_date": "2025-06-06",
        "prices": [50.0, 45.0, 60.0],
        "actions": [1.0, 0.0, -1.0],
        "storage_levels": [0.0, 1.0, 1.0, 0.0],
    }
    trajectory_path.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")

    observations, actions = load_trajectory_samples(
        trajectory_path,
        capacity=2.0,
        price_scale=50.0,
    )

    angle = 2.0 * np.pi * (157 - 1) / 365
    assert observations.shape == (3, 5)
    assert actions.shape == (3, 1)
    assert np.allclose(observations[0], [0.0, 1.0, np.sin(angle), np.cos(angle), 1.0])
    assert observations[-1, 4] == 0.0


@pytest.mark.parametrize("algorithm_name", ["ppo", "sac", "td3"])
def test_behavior_cloning_writes_policy_state_and_loads_it(
    tmp_path,
    algorithm_name: str,
) -> None:
    """Behavior cloning writes policy weights loadable by the RL runner."""
    trajectory_path = tmp_path / "perfect_foresight_trajectories_pretrain.jsonl"
    trajectory = {
        "split": "pretrain",
        "path_id": 0,
        "start_date": None,
        "end_date": None,
        "prices": [50.0, 45.0, 60.0],
        "actions": [1.0, 0.0, -1.0],
        "storage_levels": [0.0, 1.0, 1.0, 0.0],
        "objective_value": 15.0,
        "terminal_deviation": 0.0,
        "success": True,
    }
    trajectory_path.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")

    config = load_config("configs/debug.yaml")
    config["dataset_config"]["cache_dir"] = str(tmp_path / "cache")
    config["dataset_config"]["n_pretrain_paths"] = 1
    config["dataset_config"]["n_train_paths"] = 2
    config["dataset_config"]["n_validation_paths"] = 1
    config["dataset_config"]["n_test_paths"] = 1
    config["logging_config"]["run_dir"] = str(tmp_path / "runs")

    summary = run_pretraining(
        config,
        "debug",
        algorithm_name,
        trajectory_path,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
    )

    policy_path = tmp_path / "runs" / "pretraining"
    run_dirs = list(policy_path.iterdir())
    assert len(run_dirs) == 1
    assert summary["policy_state_dict"].endswith("policy_state_dict.pt")
    assert (run_dirs[0] / "policy_state_dict.pt").exists()
    assert (run_dirs[0] / "pretrained_model.zip").exists()

    dataset, params, env_kwargs = build_environment(config)
    env = GasStorageEnv(dataset, "train", params, **env_kwargs)
    model = make_sb3_agent(
        algorithm_name,
        env,
        config["agent_config"][algorithm_name],
        seed=1,
    )
    loaded_path = load_pretrained_policy_state(model, run_dirs[0])
    assert loaded_path == run_dirs[0] / "policy_state_dict.pt"
