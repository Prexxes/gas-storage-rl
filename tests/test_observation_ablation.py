"""Tests for the frozen deterministic observation-ablation diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from gas_storage_rl.training.run_observation_ablation import (
    DEFAULT_OBSERVATION_FEATURES,
    build_frozen_observation_ablation_environments,
    run_observation_ablation,
    run_standard_observation_ablation,
    solve_frozen_oracles,
)


def make_config() -> dict:
    """Returns a small observation-ablation config for tests."""
    return {
        "environment_config": {
            "episode_length": 12,
            "capacity": 10.0,
            "injection_rate": 1.0,
            "withdrawal_rate": 1.0,
            "initial_inventory": 5.0,
            "target_terminal_inventory": 5.0,
            "price_scale": 40.0,
            "reward_scale": 40.0,
            "penalty_factor": 1.0,
        },
        "observation_ablation_config": {
            "frozen_episodes": [
                {"start_date": "2020-01-01", "initial_inventory": 2.0},
                {"start_date": "2020-04-01", "initial_inventory": 5.0},
                {"start_date": "2020-07-01", "initial_inventory": 8.0},
                {"start_date": "2020-10-01", "initial_inventory": 5.0},
            ],
            "variants": {
                "full": dict(DEFAULT_OBSERVATION_FEATURES),
                "no_price": {
                    **DEFAULT_OBSERVATION_FEATURES,
                    "price": False,
                },
            },
        },
        "training_config": {
            "total_timesteps": 20,
            "eval_freq": 10,
            "n_seeds": 1,
        },
        "evaluation_config": {
            "minimum_oracle_ratio": -10.0,
            "maximum_terminal_deviation": 10.0,
            "minimum_successful_seeds": 1,
        },
        "logging_config": {"run_dir": "unused"},
        "seeds": {"master_seed": 2701},
        "agent_config": {
            "ppo": {
                "policy": "MlpPolicy",
                "gamma": 1.0,
                "n_steps": 8,
                "batch_size": 4,
                "n_epochs": 1,
                "device": "cpu",
                "verbose": 0,
            },
            "sac": {
                "policy": "MlpPolicy",
                "gamma": 1.0,
                "learning_starts": 1000,
                "buffer_size": 1000,
                "batch_size": 8,
                "device": "cpu",
                "verbose": 0,
            },
            "td3": {
                "policy": "MlpPolicy",
                "gamma": 1.0,
                "learning_starts": 1000,
                "buffer_size": 1000,
                "batch_size": 8,
                "device": "cpu",
                "verbose": 0,
            },
        },
    }


def test_frozen_environment_contains_four_deterministic_episodes() -> None:
    """Frozen benchmark has fixed paths, calendars, and inventories."""
    train_env, eval_env, _ = build_frozen_observation_ablation_environments(
        make_config(),
        seed=1,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
    )

    assert train_env.dataset.get_paths("train").shape == (4, 12)
    assert eval_env.dataset.get_paths("validation").shape == (4, 12)
    assert np.allclose(
        eval_env.dataset.get_initial_inventories("validation"),
        [2.0, 5.0, 8.0, 5.0],
    )
    obs_a, _ = eval_env.reset(options={"path_id": 0})
    obs_b, _ = eval_env.reset(options={"path_id": 1})

    assert not np.allclose(obs_a[2:4], obs_b[2:4])
    assert obs_a[0] == obs_a[5] == 0.2
    assert obs_b[0] == obs_b[5] == 0.5


def test_frozen_environment_applies_observation_mask() -> None:
    """Ablated features are neutralized while shape stays unchanged."""
    _, eval_env, _ = build_frozen_observation_ablation_environments(
        make_config(),
        seed=1,
        observation_features={
            **DEFAULT_OBSERVATION_FEATURES,
            "price": False,
            "calendar": False,
            "remaining_time": False,
            "target_inventory": False,
        },
    )

    obs, _ = eval_env.reset(options={"path_id": 0})

    assert obs.shape == (6,)
    assert obs[0] == 0.2
    assert np.allclose(obs[1:], [0.0, 0.0, 1.0, 0.0, 0.0])


def test_frozen_oracles_are_solved_for_each_episode() -> None:
    """Perfect foresight references cover every frozen validation path."""
    _, eval_env, params = build_frozen_observation_ablation_environments(
        make_config(),
        seed=1,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
    )

    oracles = solve_frozen_oracles(eval_env, params)

    assert len(oracles) == 4
    assert all(item["return_raw"] >= 0.0 for item in oracles)
    assert all(item["terminal_deviation"] <= 1e-8 for item in oracles)


@pytest.mark.parametrize("algorithm", ["ppo", "sac", "td3"])
def test_observation_ablation_smoke_writes_artifacts(tmp_path, algorithm) -> None:
    """Tiny algorithm smoke runs write summaries and evaluation curves."""
    summary = run_observation_ablation(
        make_config(),
        [algorithm],
        variants=["full"],
        output_dir=tmp_path,
    )

    output_dir = tmp_path / next(tmp_path.iterdir()).name
    assert summary["passed"]
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (
        output_dir / "full" / algorithm / "seed_0" / "evaluations.csv"
    ).exists()


def test_standard_observation_ablation_runs_groups_with_masks(
    monkeypatch,
    tmp_path,
) -> None:
    """Standard ablations forward observation masks to normal group runs."""
    config = make_config()
    config["logging_config"]["run_dir"] = str(tmp_path / "runs")
    captured = []

    def fake_run_experiment_group(config, config_name, algorithm, **kwargs):
        captured.append(
            {
                "config": config,
                "config_name": config_name,
                "algorithm": algorithm,
                "kwargs": kwargs,
            }
        )
        return {
            "group_dir": str(tmp_path / config_name),
            "completed_runs": 2,
            "skipped_runs": 0,
            "failed_runs": 0,
        }

    monkeypatch.setattr(
        "gas_storage_rl.training.run_observation_ablation.run_experiment_group",
        fake_run_experiment_group,
    )

    summary = run_standard_observation_ablation(
        config,
        "ou_c200",
        ["ppo"],
        variants=["full", "no_price"],
        seed_indices=[100, 101],
    )

    assert summary["failed_runs"] == 0
    assert summary["completed_runs"] == 4
    assert len(captured) == 2
    assert captured[0]["config_name"] == "ou_c200-full"
    assert captured[1]["config_name"] == "ou_c200-no_price"
    assert captured[0]["kwargs"]["seed_indices"] == [100, 101]
    assert captured[0]["config"]["environment_config"]["observation_features"] == {
        **DEFAULT_OBSERVATION_FEATURES,
    }
    assert captured[1]["config"]["environment_config"]["observation_features"] == {
        **DEFAULT_OBSERVATION_FEATURES,
        "price": False,
    }
    assert (tmp_path / "runs" / "observation_ablations").exists()


def test_standard_observation_ablation_can_transfer_hpo_best_config(
    monkeypatch,
    tmp_path,
) -> None:
    """HPO best settings are transferred before variant group execution."""
    config = make_config()
    config["environment_config"]["capacity"] = 200.0
    config["environment_config"]["reward_scale"] = 2.0
    config["logging_config"]["run_dir"] = str(tmp_path / "runs")
    best_config = {
        "environment_config": {"reward_scale": 8.0},
        "agent_config": {
            "ppo": {
                "policy": "MlpPolicy",
                "device": "cpu",
                "learning_rate": 0.0001,
            },
        },
    }
    captured = {}

    def fake_run_experiment_group(config, config_name, algorithm, **kwargs):
        del kwargs
        captured["config"] = config
        captured["config_name"] = config_name
        captured["algorithm"] = algorithm
        return {
            "group_dir": str(tmp_path / config_name),
            "completed_runs": 1,
            "skipped_runs": 0,
            "failed_runs": 0,
        }

    monkeypatch.setattr(
        "gas_storage_rl.training.run_observation_ablation.run_experiment_group",
        fake_run_experiment_group,
    )

    summary = run_standard_observation_ablation(
        config,
        "ou_c200",
        ["ppo"],
        variants=["no_price"],
        best_config=best_config,
    )

    assert summary["failed_runs"] == 0
    assert captured["config_name"] == "ou_c200-no_price-hpo-transfer"
    assert captured["algorithm"] == "ppo"
    assert captured["config"]["environment_config"]["capacity"] == 200.0
    assert captured["config"]["environment_config"]["reward_scale"] == 8.0
    assert captured["config"]["environment_config"]["observation_features"] == {
        **DEFAULT_OBSERVATION_FEATURES,
        "price": False,
    }
    assert captured["config"]["agent_config"]["ppo"]["learning_rate"] == 0.0001
