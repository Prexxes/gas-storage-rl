"""Tests for the deterministic overfitting diagnostic."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from gas_storage_rl.training.config import load_config
from gas_storage_rl.training.run_overfit_check import (
    EXPECTED_ORACLE_RETURN,
    OVERFIT_PRICES,
    assess_run,
    build_overfit_environments,
    run_overfit_check,
    solve_overfit_oracle,
)


def test_overfit_episode_and_oracle_have_expected_solution() -> None:
    """The fixed 20-step episode has the documented analytical optimum."""
    config = load_config("configs/sanity_overfit.yaml")
    train_env, eval_env, params = build_overfit_environments(config, seed=1)

    oracle = solve_overfit_oracle(params, eval_env.lambda_terminal)

    assert len(OVERFIT_PRICES) == 20
    assert np.isclose(oracle["return_raw"], EXPECTED_ORACLE_RETURN)
    assert np.allclose(oracle["actions"], np.tile([1.0, -1.0], 10))
    assert np.isclose(oracle["storage_levels"][-1], 0.0)
    train_env.close()
    eval_env.close()


def test_oracle_actions_reproduce_expected_environment_cashflow() -> None:
    """Executing the oracle actions produces return 203 and ends empty."""
    config = load_config("configs/sanity_overfit.yaml")
    env, eval_env, params = build_overfit_environments(config, seed=1)
    oracle = solve_overfit_oracle(params, eval_env.lambda_terminal)
    env.reset()
    infos = []

    for action in oracle["actions"]:
        _, _, terminated, truncated, info = env.step([action])
        infos.append(info)

    assert terminated
    assert not truncated
    assert np.isclose(sum(info["raw_reward"] for info in infos), 203.0)
    assert np.isclose(infos[-1]["storage_level"], 0.0)
    env.close()
    eval_env.close()


def test_assess_run_uses_oracle_ratio_and_terminal_deviation() -> None:
    """A run must satisfy both configured pass criteria."""
    passing = assess_run(
        {"mean_return_raw": 190.0, "mean_terminal_deviation": 0.0},
        oracle_return=203.0,
        minimum_oracle_ratio=0.9,
        maximum_terminal_deviation=0.01,
    )
    failing = assess_run(
        {"mean_return_raw": 203.0, "mean_terminal_deviation": 0.1},
        oracle_return=203.0,
        minimum_oracle_ratio=0.9,
        maximum_terminal_deviation=0.01,
    )

    assert passing["passed"] is True
    assert failing["passed"] is False


@pytest.mark.skipif(
    importlib.util.find_spec("stable_baselines3") is None,
    reason="stable-baselines3 is not installed",
)
@pytest.mark.parametrize("algorithm", ["ppo", "sac", "td3"])
def test_overfit_runner_smoke(tmp_path, algorithm: str) -> None:
    """Each supported agent completes a tiny diagnostic and writes artifacts."""
    config = load_config("configs/sanity_overfit.yaml")
    config["training_config"].update(
        {"total_timesteps": 20, "eval_freq": 10, "n_seeds": 1}
    )
    config["evaluation_config"].update(
        {
            "minimum_oracle_ratio": -10.0,
            "maximum_terminal_deviation": 1.0,
            "minimum_successful_seeds": 1,
        }
    )
    config["agent_config"]["ppo"].update(
        {"n_steps": 20, "batch_size": 4, "n_epochs": 1}
    )
    for off_policy_algorithm in ("sac", "td3"):
        config["agent_config"][off_policy_algorithm].update(
            {
                "learning_starts": 1,
                "batch_size": 4,
                "buffer_size": 100,
                "train_freq": 1,
                "gradient_steps": 1,
            }
        )

    summary = run_overfit_check(
        config,
        [algorithm],
        output_dir=tmp_path,
    )
    output_dir = tmp_path / next(tmp_path.iterdir()).name

    assert summary["passed"] is True
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "learning_curves.png").exists()
    assert (output_dir / algorithm / "seed_0" / "evaluations.csv").exists()
