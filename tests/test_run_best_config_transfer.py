"""Tests for transferring HPO best settings to target configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from gas_storage_rl.training import run_best_config_transfer


def test_transfer_best_agent_settings_keeps_target_environment_fields() -> None:
    """Only agent settings and reward scale are copied from the best config."""
    best_config = {
        "environment_config": {
            "environment_name": "ou",
            "capacity": 200,
            "reward_scale": 0.5,
        },
        "agent_config": {
            "ppo": {
                "policy": "MlpPolicy",
                "device": "cpu",
                "learning_rate": 0.001,
            },
        },
    }
    target_config = {
        "environment_config": {
            "environment_name": "deterministic",
            "capacity": 30,
            "reward_scale": 2.0,
        },
        "agent_config": {
            "ppo": {"policy": "MlpPolicy", "device": "cpu", "gamma": 1.0},
            "sac": {"policy": "MlpPolicy", "device": "cpu"},
        },
        "seeds": {"dataset_seed": 123},
    }

    transferred = run_best_config_transfer.transfer_best_agent_settings(
        best_config,
        target_config,
        "ppo",
    )

    assert transferred["environment_config"]["environment_name"] == "deterministic"
    assert transferred["environment_config"]["capacity"] == 30
    assert transferred["environment_config"]["reward_scale"] == 0.5
    assert transferred["agent_config"]["algorithm_name"] == "ppo"
    assert transferred["agent_config"]["ppo"]["learning_rate"] == 0.001
    assert transferred["agent_config"]["sac"] == {"policy": "MlpPolicy", "device": "cpu"}
    assert target_config["environment_config"]["reward_scale"] == 2.0


def test_run_transferred_best_config_group_uses_merged_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The runner forwards explicit seed indices to the normal group runner."""
    best_config = {
        "environment_config": {"reward_scale": 1.0},
        "agent_config": {"td3": {"policy": "MlpPolicy", "learning_rate": 0.002}},
    }
    target_config = {
        "environment_config": {"environment_name": "ou", "reward_scale": 2.0},
        "agent_config": {"td3": {"policy": "MlpPolicy"}},
    }
    captured = {}

    def fake_run_experiment_group(config, config_name, algorithm, **kwargs):
        captured["config"] = config
        captured["config_name"] = config_name
        captured["algorithm"] = algorithm
        captured["kwargs"] = kwargs
        return {"group_dir": str(tmp_path), "failed_runs": 0}

    monkeypatch.setattr(
        run_best_config_transfer,
        "run_experiment_group",
        fake_run_experiment_group,
    )

    summary = run_best_config_transfer.run_transferred_best_config_group(
        best_config,
        target_config,
        "ou_c30",
        "td3",
        seed_indices=[100, 101],
    )

    assert summary["failed_runs"] == 0
    assert captured["config_name"] == "ou_c30-hpo-transfer"
    assert captured["algorithm"] == "td3"
    assert captured["kwargs"]["seed_indices"] == [100, 101]
    assert captured["config"]["environment_config"]["environment_name"] == "ou"
    assert captured["config"]["environment_config"]["reward_scale"] == 1.0
    assert captured["config"]["agent_config"]["td3"]["learning_rate"] == 0.002


def test_write_yaml_creates_parent_directory(tmp_path: Path) -> None:
    """Merged configs can be saved for reproducible inspection."""
    output_path = tmp_path / "configs" / "merged.yaml"

    run_best_config_transfer._write_yaml(
        output_path,
        {"environment_config": {"reward_scale": 0.5}},
    )

    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == {
        "environment_config": {"reward_scale": 0.5}
    }
