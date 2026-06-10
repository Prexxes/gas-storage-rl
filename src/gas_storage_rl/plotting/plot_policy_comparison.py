"""Create per-path policy comparison diagnostics."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gas_storage_rl.baselines.oracle_cloned_policy import OracleClonedPolicy
from gas_storage_rl.baselines.perfect_foresight import PerfectForesightBaseline
from gas_storage_rl.baselines.rule_based_policy import RuleBasedPolicy
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.evaluation.run_benchmarks import LSMCPolicyAdapter
from gas_storage_rl.training.config import build_environment, load_config


def main() -> None:
    """Creates a multi-method diagnostic plot for one fixed episode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-run-dir", required=True)
    parser.add_argument("--rl-run-dir", action="append", default=[])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--path-id", type=int, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Include one random-policy trajectory using seeds.eval_seed.",
    )
    args = parser.parse_args()

    benchmark_run_dir = Path(args.benchmark_run_dir)
    config = load_config(benchmark_run_dir / "config.json")
    dataset, storage_params, env_kwargs = build_environment(config)
    trajectories = []

    for run_dir_text in args.rl_run_dir:
        run_dir = Path(run_dir_text)
        rl_config = load_config(run_dir / "config.json")
        rl_dataset, rl_storage_params, rl_env_kwargs = build_environment(rl_config)
        env = GasStorageEnv(rl_dataset, args.split, rl_storage_params, **rl_env_kwargs)
        model = _load_rl_model(
            rl_config["agent_config"]["algorithm_name"],
            run_dir / "final_model",
            env,
        )
        _, model_trajectories = evaluate_policy_on_paths(
            env,
            model,
            path_ids=[args.path_id],
            deterministic=bool(rl_config["evaluation_config"].get("deterministic", True)),
            total_training_env_steps=int(rl_config["training_config"]["total_timesteps"]),
        )
        trajectories.append(
            {
                "method": f"{rl_config['agent_config']['algorithm_name']}:{run_dir.name}",
                "infos": model_trajectories[0]["infos"],
            }
        )

    env = GasStorageEnv(dataset, args.split, storage_params, **env_kwargs)
    train_paths = dataset.get_paths("train")
    rule_policy = RuleBasedPolicy.from_training_prices(
        train_paths,
        capacity=storage_params.capacity,
        episode_length=dataset.episode_length,
        injection_rate=storage_params.injection_rate,
        withdrawal_rate=storage_params.withdrawal_rate,
    )
    trajectories.append(_evaluate_policy(env, rule_policy, "rule_based", args.path_id))

    lsmc_path = benchmark_run_dir / "lsmc_policy.pkl"
    if lsmc_path.exists():
        with lsmc_path.open("rb") as file:
            lsmc = pickle.load(file)
        trajectories.append(
            _evaluate_policy(env, LSMCPolicyAdapter(lsmc, env), "lsmc", args.path_id)
        )

    oracle_path = benchmark_run_dir / "oracle_cloned_policy.pt"
    if oracle_path.exists():
        oracle = OracleClonedPolicy.load(oracle_path)
        trajectories.append(
            _evaluate_policy(env, oracle, "oracle_cloned_policy", args.path_id)
        )

    if args.include_random:
        from gas_storage_rl.baselines.random_policy import RandomPolicy

        trajectories.append(
            _evaluate_policy(
                env,
                RandomPolicy(seed=int(config["seeds"]["eval_seed"])),
                "random",
                args.path_id,
            )
        )

    trajectories.append(
        _perfect_foresight_trajectory(
            dataset,
            args.split,
            args.path_id,
            storage_params,
            env.lambda_terminal,
        )
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else benchmark_run_dir / "plots" / "policy_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_path_{args.path_id}.png"
    _save(plot_policy_comparison(trajectories), output_path)
    print(f"Saved policy comparison plot to {output_path.resolve()}")


def plot_policy_comparison(trajectories: list[dict[str, Any]]) -> plt.Figure:
    """Plots price, actions, storage, and cashflow for multiple methods."""
    figure, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    base_infos = trajectories[0]["infos"]
    steps = np.array([info["current_step"] for info in base_infos])
    prices = np.array([info["price"] for info in base_infos])
    axes[0].plot(steps, prices, color="black", label="spot price")
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8)

    for trajectory in trajectories:
        label = trajectory["method"]
        infos = trajectory["infos"]
        x_values = np.array([info["current_step"] for info in infos])
        actions = np.array([info["executed_action"] for info in infos])
        storage = np.array([info["storage_level"] for info in infos])
        cashflow = np.cumsum([info["raw_cashflow"] for info in infos])
        axes[1].step(x_values, actions, where="post", label=label)
        axes[2].plot(x_values, storage, label=label)
        axes[3].plot(x_values, cashflow, label=label)

    axes[1].set_ylabel("Action")
    axes[2].set_ylabel("Storage")
    axes[3].set_ylabel("Cumulative cashflow")
    axes[3].set_xlabel("Episode timestep")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    return figure


def _evaluate_policy(
    env: GasStorageEnv,
    policy: Any,
    method: str,
    path_id: int,
) -> dict[str, Any]:
    """Evaluates one policy on one fixed path and labels the trajectory."""
    _, trajectories = evaluate_policy_on_paths(env, policy, path_ids=[path_id])
    return {"method": method, "infos": trajectories[0]["infos"]}


def _perfect_foresight_trajectory(
    dataset: Any,
    split: str,
    path_id: int,
    storage_params: Any,
    lambda_terminal: float,
) -> dict[str, Any]:
    """Returns a step trajectory from the perfect-foresight solution."""
    path = dataset.get_path(split, path_id)
    initial = float(dataset.get_initial_inventories(split, storage_params.initial_inventory)[path_id])
    result = PerfectForesightBaseline(storage_params, lambda_terminal).solve_path(
        path,
        initial,
        initial,
    )
    infos = []
    cumulative_storage = result.storage_levels
    for step, (price, action) in enumerate(zip(path, result.actions, strict=True)):
        storage_level = float(cumulative_storage[step + 1])
        raw_cashflow = -float(action) * float(price)
        terminal_penalty = 0.0
        if step == len(path) - 1:
            terminal_penalty = -lambda_terminal * abs(storage_level - initial)
        infos.append(
            {
                "current_step": step,
                "price": float(price),
                "requested_action": float(action),
                "executed_action": float(action),
                "storage_level": storage_level,
                "raw_cashflow": raw_cashflow,
                "terminal_penalty": terminal_penalty,
                "start_index": int(dataset.get_start_indices(split)[path_id]),
                "initial_inventory": initial,
                "target_terminal_inventory": initial,
                "path_id": path_id,
                "split": split,
            }
        )
    return {"method": "perfect_foresight", "infos": infos}


def _load_rl_model(algorithm_name: str, model_path: Path, env: GasStorageEnv) -> Any:
    """Loads a saved Stable-Baselines3 model."""
    if algorithm_name == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(model_path, env=env, device="cpu")
    if algorithm_name == "sac":
        from stable_baselines3 import SAC

        return SAC.load(model_path, env=env, device="cpu")
    if algorithm_name == "td3":
        from stable_baselines3 import TD3

        return TD3.load(model_path, env=env, device="cpu")
    raise ValueError(f"Unsupported algorithm_name: {algorithm_name}")


def _save(figure: plt.Figure, path: Path) -> None:
    """Saves and closes a Matplotlib figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
