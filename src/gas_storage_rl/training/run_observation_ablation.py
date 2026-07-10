"""Observation-feature ablation on a frozen deterministic storage benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gas_storage_rl.agents.sb3_factory import make_sb3_agent
from gas_storage_rl.baselines.perfect_foresight import PerfectForesightBaseline
from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.gas_storage_env import DEFAULT_OBSERVATION_FEATURES
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.envs.storage_dynamics import StorageParams
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.training.config import load_config
from gas_storage_rl.training.run_overfit_check import (
    OverfitEvaluationCallback,
    SUPPORTED_ALGORITHMS,
    _create_output_dir,
    _seed_for_run,
    _validate_algorithms,
    _write_csv,
    _write_json,
)


DEFAULT_FROZEN_EPISODES = [
    {"start_date": "2020-01-01", "initial_inventory": 2.0},
    {"start_date": "2020-04-01", "initial_inventory": 5.0},
    {"start_date": "2020-07-01", "initial_inventory": 8.0},
    {"start_date": "2020-10-01", "initial_inventory": 5.0},
]

DEFAULT_VARIANTS = {
    "full": {
        "inventory": True,
        "price": True,
        "calendar": True,
        "remaining_time": True,
        "target_inventory": True,
    },
    "price_inventory_only": {
        "inventory": True,
        "price": True,
        "calendar": False,
        "remaining_time": False,
        "target_inventory": False,
    },
    "no_calendar": {
        "inventory": True,
        "price": True,
        "calendar": False,
        "remaining_time": True,
        "target_inventory": True,
    },
    "no_remaining_time": {
        "inventory": True,
        "price": True,
        "calendar": True,
        "remaining_time": False,
        "target_inventory": True,
    },
    "no_target_inventory": {
        "inventory": True,
        "price": True,
        "calendar": True,
        "remaining_time": True,
        "target_inventory": False,
    },
    "no_price": {
        "inventory": True,
        "price": False,
        "calendar": True,
        "remaining_time": True,
        "target_inventory": True,
    },
    "no_inventory": {
        "inventory": False,
        "price": True,
        "calendar": True,
        "remaining_time": True,
        "target_inventory": True,
    },
}


def build_frozen_observation_ablation_environments(
    config: dict[str, Any],
    seed: int,
    observation_features: dict[str, bool] | None = None,
) -> tuple[GasStorageEnv, GasStorageEnv, StorageParams]:
    """Builds train and validation environments from fixed deterministic episodes."""
    ablation_config = config.get("observation_ablation_config", {})
    env_config = config["environment_config"]
    episode_length = int(env_config["episode_length"])
    frozen_episodes = ablation_config.get(
        "frozen_episodes",
        DEFAULT_FROZEN_EPISODES,
    )
    paths = np.asarray(
        [
            _deterministic_price_path(item["start_date"], episode_length)
            for item in frozen_episodes
        ],
        dtype=np.float32,
    )
    date_ranges = [
        {
            "start_date": item["start_date"],
            "end_date": (
                datetime.strptime(item["start_date"], "%Y-%m-%d").date()
                + timedelta(days=episode_length - 1)
            ).isoformat(),
        }
        for item in frozen_episodes
    ]
    initial_inventories = np.asarray(
        [float(item["initial_inventory"]) for item in frozen_episodes],
        dtype=np.float64,
    )
    dataset = PathDataset(
        {"train": paths.copy(), "validation": paths.copy()},
        {"train": seed, "validation": seed},
        date_ranges_by_split={
            "train": date_ranges,
            "validation": date_ranges,
        },
        initial_inventories_by_split={
            "train": initial_inventories.copy(),
            "validation": initial_inventories.copy(),
        },
    )
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
        "observation_features": observation_features,
        "seed": seed,
    }
    return (
        GasStorageEnv(dataset, "train", params, **env_kwargs),
        GasStorageEnv(dataset, "validation", params, **env_kwargs),
        params,
    )


def solve_frozen_oracles(
    env: GasStorageEnv,
    params: StorageParams,
) -> list[dict[str, Any]]:
    """Solves perfect-foresight references for all frozen validation episodes."""
    terminal_lambda = max(env.lambda_terminal, 1_000_000.0)
    baseline = PerfectForesightBaseline(params, terminal_lambda)
    oracles = []
    for path_id, prices in enumerate(env.dataset.get_paths(env.split)):
        initial_inventory = float(
            env.dataset.get_initial_inventories(env.split)[path_id]
        )
        result = baseline.solve_path(
            prices,
            initial_inventory=initial_inventory,
            target_inventory=initial_inventory,
        )
        if not result.success:
            raise RuntimeError(f"Perfect-foresight optimization failed for {path_id}")
        oracles.append(
            {
                "path_id": path_id,
                "return_raw": float(result.objective_value),
                "terminal_deviation": float(result.terminal_deviation),
                "initial_inventory": initial_inventory,
            }
        )
    return oracles


def run_observation_ablation(
    config: dict[str, Any],
    algorithms: list[str],
    *,
    variants: list[str] | None = None,
    total_timesteps: int | None = None,
    n_seeds: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Runs observation-feature ablations on the frozen deterministic benchmark."""
    _validate_algorithms(algorithms)
    training_config = config["training_config"]
    evaluation_config = config["evaluation_config"]
    ablation_config = config.get("observation_ablation_config", {})
    variant_definitions = _variant_definitions(ablation_config)
    selected_variants = variants or list(variant_definitions)
    unknown_variants = sorted(set(selected_variants) - set(variant_definitions))
    if unknown_variants:
        raise ValueError(f"Unknown variants: {', '.join(unknown_variants)}")

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
    _, reference_env, params = build_frozen_observation_ablation_environments(
        config,
        master_seed,
        DEFAULT_OBSERVATION_FEATURES,
    )
    oracles = solve_frozen_oracles(reference_env, params)
    oracle_return = float(np.mean([item["return_raw"] for item in oracles]))
    oracle = {"mean_return_raw": oracle_return, "episodes": oracles}
    _write_json(group_dir / "oracle.json", oracle)

    runs = []
    for variant in selected_variants:
        observation_features = variant_definitions[variant]
        for algorithm in algorithms:
            for seed_index in range(seed_count):
                seed = _seed_for_run(master_seed, seed_index)
                run_dir = group_dir / variant / algorithm / f"seed_{seed_index}"
                run_dir.mkdir(parents=True)
                train_env, eval_env, _ = build_frozen_observation_ablation_environments(
                    config,
                    seed,
                    observation_features,
                )
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
                terminal_deviation = float(final_metrics["mean_terminal_deviation"])
                oracle_ratio = float(final_metrics["mean_return_raw"]) / oracle_return
                run_summary = {
                    "variant": variant,
                    "algorithm": algorithm,
                    "seed_index": seed_index,
                    "seed": seed,
                    "requested_training_steps": total_steps,
                    "actual_training_steps": int(model.num_timesteps),
                    "training_wall_time_s": time.time() - started,
                    "final_return_raw": float(final_metrics["mean_return_raw"]),
                    "oracle_ratio": oracle_ratio,
                    "terminal_deviation": terminal_deviation,
                    "terminal_penalty": float(final_metrics["mean_terminal_penalty"]),
                    "constrained_actions": float(
                        final_metrics["mean_number_of_constrained_actions"]
                    ),
                    "terminal_feasibility_clipped_actions": (
                        _mean_terminal_feasibility_clipped_actions(trajectories)
                    ),
                    "passed": bool(
                        oracle_ratio
                        >= float(evaluation_config["minimum_oracle_ratio"])
                        and terminal_deviation
                        <= float(evaluation_config["maximum_terminal_deviation"])
                    ),
                }
                _write_json(run_dir / "summary.json", run_summary)
                model.save(run_dir / "final_model")
                train_env.close()
                eval_env.close()
                runs.append(run_summary)

    summary = _aggregate_ablation_summary(
        runs,
        selected_variants,
        algorithms,
        seed_count,
        evaluation_config,
        oracle,
    )
    _write_json(group_dir / "summary.json", summary)
    _write_csv(group_dir / "summary.csv", runs)
    _plot_ablation_learning_curves(group_dir, runs, oracle_return)
    reference_env.close()
    return {"output_dir": str(group_dir), **summary}


def _deterministic_price_path(start_date: str, episode_length: int) -> np.ndarray:
    """Returns one fixed seasonal price path with deterministic local events."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    prices = []
    event_shocks = {15: -8.0, 35: 12.0, 60: -7.0, 75: 10.0}
    for step in range(episode_length):
        current = start + timedelta(days=step)
        day_of_year = current.timetuple().tm_yday - 1
        days_in_year = 366 if _is_leap_year(current.year) else 365
        seasonal = 10.0 * np.cos(2.0 * np.pi * day_of_year / days_in_year)
        short_cycle = 3.0 * np.sin(2.0 * np.pi * step / 14.0)
        prices.append(30.0 + seasonal + short_cycle + event_shocks.get(step, 0.0))
    return np.asarray(prices, dtype=np.float32)


def _variant_definitions(
    ablation_config: dict[str, Any],
) -> dict[str, dict[str, bool]]:
    variants = ablation_config.get("variants", DEFAULT_VARIANTS)
    return {
        name: {**DEFAULT_OBSERVATION_FEATURES, **dict(mask)}
        for name, mask in variants.items()
    }


def _mean_terminal_feasibility_clipped_actions(
    trajectories: list[dict[str, Any]],
) -> float:
    counts = [
        sum(bool(info["terminal_feasibility_clipped"]) for info in trajectory["infos"])
        for trajectory in trajectories
    ]
    return float(np.mean(counts))


def _aggregate_ablation_summary(
    runs: list[dict[str, Any]],
    variants: list[str],
    algorithms: list[str],
    n_seeds: int,
    evaluation_config: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    configured_minimum = int(
        evaluation_config.get("minimum_successful_seeds", min(2, n_seeds))
    )
    minimum_successes = min(configured_minimum, n_seeds)
    groups = {}
    for variant in variants:
        groups[variant] = {}
        for algorithm in algorithms:
            group_runs = [
                run
                for run in runs
                if run["variant"] == variant and run["algorithm"] == algorithm
            ]
            passed_seeds = sum(bool(run["passed"]) for run in group_runs)
            groups[variant][algorithm] = {
                "passed_seeds": passed_seeds,
                "n_seeds": n_seeds,
                "mean_final_return_raw": float(
                    np.mean([run["final_return_raw"] for run in group_runs])
                ),
                "mean_oracle_ratio": float(
                    np.mean([run["oracle_ratio"] for run in group_runs])
                ),
                "mean_terminal_deviation": float(
                    np.mean([run["terminal_deviation"] for run in group_runs])
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
        "variants": groups,
        "passed": all(
            item["passed"]
            for variant_summary in groups.values()
            for item in variant_summary.values()
        ),
        "runs": runs,
    }


def _plot_ablation_learning_curves(
    group_dir: Path,
    runs: list[dict[str, Any]],
    oracle_return: float,
) -> None:
    figure, axis = plt.subplots()
    for run in runs:
        evaluation_path = (
            group_dir
            / run["variant"]
            / run["algorithm"]
            / f"seed_{run['seed_index']}"
            / "evaluations.csv"
        )
        with evaluation_path.open("r", encoding="utf-8", newline="") as file:
            import csv

            rows = list(csv.DictReader(file))
        steps = [int(row["total_training_env_steps"]) for row in rows]
        ratios = [float(row["mean_return_raw"]) / oracle_return for row in rows]
        axis.plot(
            steps,
            ratios,
            alpha=0.7,
            label=(
                f"{run['variant']} {run['algorithm']} "
                f"seed {run['seed_index']}"
            ),
        )
    axis.axhline(1.0, color="gray", linestyle=":", label="oracle")
    axis.set_xlabel("Total environment steps")
    axis.set_ylabel("Deterministic return / mean oracle return")
    axis.legend(fontsize="x-small")
    figure.tight_layout()
    figure.savefig(group_dir / "learning_curves.png", dpi=150)
    plt.close(figure)


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def main() -> None:
    """Runs the observation ablation diagnostic from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sanity_observation_ablation.yaml")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=SUPPORTED_ALGORITHMS,
        default=list(SUPPORTED_ALGORITHMS),
    )
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = run_observation_ablation(
        load_config(args.config),
        args.algorithms,
        variants=args.variants,
        total_timesteps=args.total_timesteps,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
