"""Tests for single-run holdout evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.evaluation import run_holdout_evaluation


def test_model_path_defaults_to_best_validation_checkpoint(tmp_path: Path) -> None:
    """Holdout evaluation resolves best_validation_model.zip by default."""
    (tmp_path / "best_validation_model.zip").write_text("model", encoding="utf-8")

    path = run_holdout_evaluation._model_path(tmp_path, "best_validation")

    assert path == tmp_path / "best_validation_model"


def test_model_device_reads_original_and_effective_config_shapes() -> None:
    """Device lookup supports YAML configs and persisted effective run configs."""
    assert (
        run_holdout_evaluation._model_device(
            {"agent_config": {"ppo": {"device": "cpu"}}},
            "ppo",
        )
        == "cpu"
    )
    assert (
        run_holdout_evaluation._model_device(
            {"agent_config": {"hyperparameters": {"device": "cpu"}}},
            "ppo",
        )
        == "cpu"
    )
    assert (
        run_holdout_evaluation._model_device(
            {"agent_config": {"effective_hyperparameters": {"device": "cpu"}}},
            "ppo",
        )
        == "cpu"
    )


def test_episode_rows_include_perfect_foresight_reference_metrics() -> None:
    """Per-run episode rows carry policy outcomes and path-wise oracle metrics."""
    dataset = PathDataset(
        {
            "train": [[10.0, 20.0]],
            "test": [[10.0, 20.0]],
        },
        {"train": 1, "test": 2},
        {
            "test": [
                {"start_date": "2024-01-01", "end_date": "2024-01-02"},
            ],
        },
    )
    trajectories = [
        {
            "path_id": 0,
            "infos": [
                {
                    "start_index": 0,
                    "initial_inventory": 0.0,
                    "target_terminal_inventory": 0.0,
                    "raw_cashflow": 10.0,
                    "terminal_penalty": 0.0,
                    "raw_reward": 10.0,
                    "scaled_reward": 1.0,
                    "requested_action": -1.0,
                    "rate_capacity_clipped_action": -1.0,
                    "executed_action": -1.0,
                    "terminal_feasibility_clipped": False,
                    "storage_level": 0.0,
                }
            ],
        }
    ]
    rows = run_holdout_evaluation._episode_rows(
        trajectories,
        method="ppo",
        split="test",
        dataset=dataset,
        seed=123,
        perfect_foresight_references=[
            {
                "path_id": 0,
                "episode_perfect_foresight_return_raw": 20.0,
            }
        ],
    )
    metrics = {}
    run_holdout_evaluation._add_holdout_reference_metrics(
        metrics,
        rows,
        n_bootstrap=100,
        bootstrap_seed=0,
    )

    assert rows[0]["episode_return_raw"] == 10.0
    assert rows[0]["episode_perfect_foresight_return_raw"] == 20.0
    assert rows[0]["episode_perfect_foresight_ratio"] == 0.5
    assert rows[0]["episode_optimality_gap"] == 0.5
    assert metrics["mean_perfect_foresight_ratio"] == 0.5
    assert metrics["mean_optimality_gap"] == 0.5


def test_load_model_passes_device_to_sb3_loader(monkeypatch, tmp_path: Path) -> None:
    """Holdout model loading honors the device from the run config."""
    calls = []

    class FakePPO:
        @staticmethod
        def load(model_path, env, device="auto"):
            calls.append((model_path, env, device))
            return "model"

    monkeypatch.setitem(
        __import__("sys").modules,
        "stable_baselines3",
        SimpleNamespace(PPO=FakePPO),
    )

    model = run_holdout_evaluation._load_model(
        "ppo",
        tmp_path / "best_validation_model",
        env="env",
        device="cpu",
    )

    assert model == "model"
    assert calls == [(tmp_path / "best_validation_model", "env", "cpu")]
