"""Intentional overfitting diagnostic on one deterministic storage episode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from gas_storage_rl.agents.sb3_factory import make_sb3_agent
from gas_storage_rl.baselines.perfect_foresight import PerfectForesightBaseline
from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.envs.storage_dynamics import StorageParams
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.training.config import load_config


OVERFIT_PRICES = np.asarray(
    [
        10.0,
        30.0,
        12.0,
        28.0,
        8.0,
        35.0,
        15.0,
        25.0,
        9.0,
        32.0,
        11.0,
        27.0,
        7.0,
        40.0,
        14.0,
        24.0,
        6.0,
        38.0,
        13.0,
        29.0,
    ],
    dtype=np.float32,
)
EXPECTED_ORACLE_RETURN = 203.0
SUPPORTED_ALGORITHMS = ("ppo", "sac", "td3")


def build_overfit_environments(
    config: dict[str, Any],
    seed: int,
) -> tuple[GasStorageEnv, GasStorageEnv, StorageParams]:
    """Builds train and evaluation environments containing the same fixed path."""
    prices = OVERFIT_PRICES.reshape(1, -1)
    dataset = PathDataset(
        {"train": prices.copy(), "validation": prices.copy()},
        {"train": seed, "validation": seed},
    )
    env_config = config["environment_config"]
    params = StorageParams(
        capacity=float(env_config["capacity"]),
        injection_rate=float(env_config["injection_rate"]),
        withdrawal_rate=float(env_config["withdrawal_rate"]),
        initial_inventory=float(env_config["initial_inventory"]),
        target_terminal_inventory=float(env_config["target_terminal_inventory"]),
    )
    env_kwargs = {
        "price_scale": float(env_config["price_scale"]),
        "reward_scale": float(env_config["reward_scale"]),
        "penalty_factor": float(env_config["penalty_factor"]),
        "fixed_path_id": 0,
        "seed": seed,
    }
    return (
        GasStorageEnv(dataset, "train", params, **env_kwargs),
        GasStorageEnv(dataset, "validation", params, **env_kwargs),
        params,
    )


def solve_overfit_oracle(
    params: StorageParams,
    lambda_terminal: float,
) -> dict[str, Any]:
    """Solves and validates the perfect-foresight reference episode."""
    result = PerfectForesightBaseline(params, lambda_terminal).solve_path(
        OVERFIT_PRICES,
        initial_inventory=params.initial_inventory,
        target_inventory=params.target_terminal_inventory,
    )
    if not result.success:
        raise RuntimeError("Perfect-foresight optimization failed")
    if not np.isclose(result.objective_value, EXPECTED_ORACLE_RETURN):
        raise RuntimeError(
            "Unexpected perfect-foresight return: "
            f"{result.objective_value}, expected {EXPECTED_ORACLE_RETURN}"
        )
    return {
        "return_raw": float(result.objective_value),
        "actions": result.actions.astype(float).tolist(),
        "storage_levels": result.storage_levels.astype(float).tolist(),
        "terminal_deviation": float(result.terminal_deviation),
    }


def assess_run(
    final_metrics: dict[str, Any],
    oracle_return: float,
    minimum_oracle_ratio: float,
    maximum_terminal_deviation: float,
) -> dict[str, Any]:
    """Computes the pass/fail result for one algorithm seed."""
    final_return = float(final_metrics["mean_return_raw"])
    oracle_ratio = final_return / oracle_return
    terminal_deviation = float(final_metrics["mean_terminal_deviation"])
    return {
        "final_return_raw": final_return,
        "oracle_ratio": oracle_ratio,
        "terminal_deviation": terminal_deviation,
        "passed": bool(
            oracle_ratio >= minimum_oracle_ratio
            and terminal_deviation <= maximum_terminal_deviation
        ),
    }


class OverfitEvaluationCallback(BaseCallback):
    """Evaluates deterministically on the fixed episode during training."""

    def __init__(
        self,
        eval_env: GasStorageEnv,
        eval_freq: int,
        output_dir: Path,
    ) -> None:
        """Initializes the periodic evaluator."""
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.output_dir = output_dir
        self.rows: list[dict[str, Any]] = []
        self.best_return = float("-inf")
        self.last_evaluation_step: int | None = None

    def _on_training_start(self) -> None:
        self.evaluate()

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self.evaluate()
        return True

    def evaluate(self) -> dict[str, Any]:
        """Runs and records one update-free deterministic evaluation episode."""
        metrics, _ = evaluate_policy_on_paths(
            self.eval_env,
            self.model,
            deterministic=True,
            total_training_env_steps=self.num_timesteps,
        )
        self.rows.append(metrics)
        self.last_evaluation_step = self.num_timesteps
        _write_csv(self.output_dir / "evaluations.csv", self.rows)
        current_return = float(metrics["mean_return_raw"])
        if current_return > self.best_return:
            self.best_return = current_return
            self.model.save(self.output_dir / "best_model")
        return metrics


def run_overfit_check(
    config: dict[str, Any],
    algorithms: list[str],
    *,
    total_timesteps: int | None = None,
    n_seeds: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Runs the deterministic memorization diagnostic."""
    _validate_algorithms(algorithms)
    training_config = config["training_config"]
    evaluation_config = config["evaluation_config"]
    total_steps = int(
        training_config["total_timesteps"]
        if total_timesteps is None
        else total_timesteps
    )
    seed_count = int(training_config["n_seeds"] if n_seeds is None else n_seeds)
    if total_steps <= 0 or seed_count <= 0:
        raise ValueError("total_timesteps and n_seeds must be positive")

    group_dir = _create_output_dir(
        output_dir or config["logging_config"]["run_dir"]
    )
    _write_json(group_dir / "config.json", config)
    master_seed = int(config["seeds"]["master_seed"])
    minimum_ratio = float(evaluation_config["minimum_oracle_ratio"])
    maximum_deviation = float(evaluation_config["maximum_terminal_deviation"])

    _, reference_env, params = build_overfit_environments(config, master_seed)
    oracle = solve_overfit_oracle(params, reference_env.lambda_terminal)
    _write_json(group_dir / "oracle.json", oracle)

    runs = []
    for algorithm in algorithms:
        for seed_index in range(seed_count):
            seed = _seed_for_run(master_seed, seed_index)
            run_dir = group_dir / algorithm / f"seed_{seed_index}"
            run_dir.mkdir(parents=True)
            train_env, eval_env, _ = build_overfit_environments(config, seed)
            model = make_sb3_agent(
                algorithm,
                train_env,
                config["agent_config"][algorithm],
                seed=seed,
            )
            callback = OverfitEvaluationCallback(
                eval_env,
                int(training_config["eval_freq"]),
                run_dir,
            )
            started = time.time()
            model.learn(total_timesteps=total_steps, callback=callback)
            if callback.last_evaluation_step != model.num_timesteps:
                callback.evaluate()
            final_metrics, trajectories = evaluate_policy_on_paths(
                eval_env,
                model,
                deterministic=True,
                total_training_env_steps=model.num_timesteps,
            )
            assessment = assess_run(
                final_metrics,
                float(oracle["return_raw"]),
                minimum_ratio,
                maximum_deviation,
            )
            infos = trajectories[0]["infos"]
            first_success_step = _first_success_step(
                callback.rows,
                float(oracle["return_raw"]),
                minimum_ratio,
            )
            run_summary = {
                "algorithm": algorithm,
                "seed_index": seed_index,
                "seed": seed,
                "requested_training_steps": total_steps,
                "actual_training_steps": int(model.num_timesteps),
                "training_wall_time_s": time.time() - started,
                "first_success_step": first_success_step,
                **assessment,
                "requested_actions": [
                    float(info["requested_action"]) for info in infos
                ],
                "executed_actions": [
                    float(info["executed_action"]) for info in infos
                ],
                "storage_levels": [float(info["storage_level"]) for info in infos],
                "cashflows": [float(info["raw_cashflow"]) for info in infos],
                "terminal_penalty": float(final_metrics["mean_terminal_penalty"]),
            }
            _write_json(run_dir / "summary.json", run_summary)
            model.save(run_dir / "final_model")
            train_env.close()
            eval_env.close()
            runs.append(run_summary)

    summary = _aggregate_summary(
        runs,
        algorithms,
        seed_count,
        oracle,
        evaluation_config,
    )
    _write_json(group_dir / "summary.json", summary)
    _write_csv(group_dir / "summary.csv", runs)
    _plot_learning_curves(
        group_dir,
        runs,
        float(oracle["return_raw"]),
        minimum_ratio,
    )
    reference_env.close()
    return {"output_dir": str(group_dir), **summary}


def _aggregate_summary(
    runs: list[dict[str, Any]],
    algorithms: list[str],
    n_seeds: int,
    oracle: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    configured_minimum = int(
        evaluation_config.get("minimum_successful_seeds", min(2, n_seeds))
    )
    minimum_successes = min(configured_minimum, n_seeds)
    if not 1 <= minimum_successes <= n_seeds:
        raise ValueError("minimum_successful_seeds must be between 1 and n_seeds")
    algorithm_summaries = {}
    for algorithm in algorithms:
        algorithm_runs = [run for run in runs if run["algorithm"] == algorithm]
        passed_seeds = sum(bool(run["passed"]) for run in algorithm_runs)
        algorithm_summaries[algorithm] = {
            "passed_seeds": passed_seeds,
            "n_seeds": n_seeds,
            "mean_final_return_raw": float(
                np.mean([run["final_return_raw"] for run in algorithm_runs])
            ),
            "mean_oracle_ratio": float(
                np.mean([run["oracle_ratio"] for run in algorithm_runs])
            ),
            "passed": passed_seeds >= minimum_successes,
        }
    return {
        "oracle": oracle,
        "minimum_oracle_ratio": float(
            evaluation_config["minimum_oracle_ratio"]
        ),
        "maximum_terminal_deviation": float(
            evaluation_config["maximum_terminal_deviation"]
        ),
        "minimum_successful_seeds": minimum_successes,
        "algorithms": algorithm_summaries,
        "passed": all(item["passed"] for item in algorithm_summaries.values()),
        "runs": runs,
    }


def _first_success_step(
    rows: list[dict[str, Any]],
    oracle_return: float,
    minimum_ratio: float,
) -> int | None:
    for row in rows:
        if float(row["mean_return_raw"]) / oracle_return >= minimum_ratio:
            return int(row["total_training_env_steps"])
    return None


def _seed_for_run(master_seed: int, seed_index: int) -> int:
    sequence = np.random.SeedSequence([master_seed, seed_index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _create_output_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output = base / timestamp
    suffix = 1
    while output.exists():
        output = base / f"{timestamp}-{suffix}"
        suffix += 1
    output.mkdir(parents=True)
    return output


def _plot_learning_curves(
    group_dir: Path,
    runs: list[dict[str, Any]],
    oracle_return: float,
    minimum_oracle_ratio: float,
) -> None:
    figure, axis = plt.subplots()
    for run in runs:
        evaluation_path = (
            group_dir
            / run["algorithm"]
            / f"seed_{run['seed_index']}"
            / "evaluations.csv"
        )
        with evaluation_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        steps = [int(row["total_training_env_steps"]) for row in rows]
        ratios = [float(row["mean_return_raw"]) / oracle_return for row in rows]
        axis.plot(
            steps,
            ratios,
            alpha=0.7,
            label=f"{run['algorithm']} seed {run['seed_index']}",
        )
    axis.axhline(
        minimum_oracle_ratio,
        color="black",
        linestyle="--",
        label=f"{minimum_oracle_ratio:.0%} oracle",
    )
    axis.axhline(1.0, color="gray", linestyle=":", label="oracle")
    axis.set_xlabel("Total environment steps")
    axis.set_ylabel("Deterministic return / oracle return")
    axis.legend()
    figure.tight_layout()
    figure.savefig(group_dir / "learning_curves.png", dpi=150)
    plt.close(figure)


def _validate_algorithms(algorithms: list[str]) -> None:
    invalid = sorted(set(algorithms) - set(SUPPORTED_ALGORITHMS))
    if invalid:
        raise ValueError(f"Unsupported algorithms: {', '.join(invalid)}")
    if not algorithms:
        raise ValueError("At least one algorithm is required")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Runs the overfitting diagnostic from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sanity_overfit.yaml")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=SUPPORTED_ALGORITHMS,
        default=list(SUPPORTED_ALGORITHMS),
    )
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = run_overfit_check(
        load_config(args.config),
        args.algorithms,
        total_timesteps=args.total_timesteps,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
