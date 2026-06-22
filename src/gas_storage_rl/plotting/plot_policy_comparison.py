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
from matplotlib.lines import Line2D

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
    """Plots price, actions, storage, and economic return for multiple methods."""
    figure = plt.figure(figsize=(12, 9), layout="constrained")
    grid = figure.add_gridspec(
        4,
        2,
        width_ratios=[1.0, 0.025],
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.14,
        wspace=0.04,
    )
    axes = [figure.add_subplot(grid[0, 0])]
    axes.extend(
        figure.add_subplot(grid[row, 0], sharex=axes[0])
        for row in range(1, 4)
    )
    colorbar_axis = figure.add_subplot(grid[1, 1])
    base_infos = trajectories[0]["infos"]
    steps = np.array([info["current_step"] for info in base_infos])
    prices = np.array([info["price"] for info in base_infos])
    axes[0].plot(steps, prices, color="black", label="spot price")
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8)

    action_matrix = np.array(
        [
            action
            for trajectory in trajectories
            for action in (
                [
                    info.get("requested_action", info["executed_action"])
                    for info in trajectory["infos"]
                ],
                [info["executed_action"] for info in trajectory["infos"]],
            )
        ],
        dtype=np.float64,
    )
    step_edges = np.arange(len(steps) + 1, dtype=np.float64) - 0.5
    action_edges = np.arange(2 * len(trajectories) + 1, dtype=np.float64) / 2.0
    action_heatmap = axes[1].pcolormesh(
        step_edges,
        action_edges,
        action_matrix,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        shading="flat",
    )
    axes[1].set_yticks(
        np.arange(len(trajectories), dtype=np.float64) + 0.5,
        [trajectory["method"] for trajectory in trajectories],
    )
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Method")
    clipped_action_boundaries = [
        (float(trajectory_index) + 0.5, float(info["current_step"]) - 0.5)
        for trajectory_index, trajectory in enumerate(trajectories)
        for info in trajectory["infos"]
        if abs(
            float(info.get("requested_action", info["executed_action"]))
            - float(info["executed_action"])
        )
        > 1e-8
    ]
    if clipped_action_boundaries:
        boundary_y_values, boundary_x_min_values = zip(*clipped_action_boundaries)
        axes[1].hlines(
            boundary_y_values,
            boundary_x_min_values,
            np.asarray(boundary_x_min_values) + 1.0,
            colors="gold",
            linewidths=3.0,
        )
    axes[1].legend(
        handles=[Line2D([0], [0], color="gold", linewidth=3.0, label="Action clipped")],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
        fontsize=7,
        framealpha=0.9,
    )
    figure.colorbar(
        action_heatmap,
        cax=colorbar_axis,
        label="Action (top: requested, bottom: executed)",
    )

    for trajectory in trajectories:
        label = trajectory["method"]
        infos = trajectory["infos"]
        x_values = np.array([info["current_step"] for info in infos])
        storage = np.array([info["storage_level"] for info in infos])
        economic_rewards = np.array(
            [
                info.get(
                    "economic_reward_raw",
                    info["raw_cashflow"] + info.get("terminal_penalty", 0.0),
                )
                for info in infos
            ],
            dtype=np.float64,
        )
        cumulative_return = np.cumsum(economic_rewards)
        axes[2].plot(x_values, storage, label=label)
        axes[3].plot(x_values, cumulative_return, label=label)

    axes[2].set_ylabel("Storage")
    axes[3].set_ylabel("Cumulative raw return")
    axes[3].set_xlabel("Episode timestep")
    for axis in (axes[0], axes[2], axes[3]):
        axis.grid(alpha=0.25)
    axes[2].legend(fontsize=7, ncol=2)
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
        economic_reward_raw = raw_cashflow + terminal_penalty
        infos.append(
            {
                "current_step": step,
                "price": float(price),
                "requested_action": float(action),
                "executed_action": float(action),
                "storage_level": storage_level,
                "raw_cashflow": raw_cashflow,
                "terminal_penalty": terminal_penalty,
                "economic_reward_raw": economic_reward_raw,
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
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
