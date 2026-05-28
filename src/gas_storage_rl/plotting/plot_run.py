"""Create trajectory plots for a saved experiment run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.plotting.policy_plots import (
    plot_agent_actions,
    plot_cumulative_cashflows,
    plot_price_action_scatter,
    plot_storage_levels,
)
from gas_storage_rl.plotting.price_plots import plot_price_paths
from gas_storage_rl.training.config import build_environment, load_config


def main() -> None:
    """Creates standard plots for one saved run and split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Run directory under runs/.")
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to plot.",
    )
    parser.add_argument(
        "--path-id",
        type=int,
        default=0,
        help="First path id to evaluate and plot.",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=1,
        help="Number of consecutive paths to evaluate and plot.",
    )
    parser.add_argument(
        "--model",
        default="final",
        choices=["final", "best"],
        help="Use final_model.zip or best_validation_model.zip.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic actions.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir / "config.json")
    dataset, storage_params, env_kwargs = build_environment(config)
    env = GasStorageEnv(dataset, args.split, storage_params, **env_kwargs)
    model = _load_model(
        algorithm_name=config["agent_config"]["algorithm_name"],
        model_path=_model_path(run_dir, args.model),
        env=env,
    )

    split_paths = dataset.get_paths(args.split)
    path_ids = list(
        range(args.path_id, min(args.path_id + args.n_paths, len(split_paths)))
    )
    metrics, trajectories = evaluate_policy_on_paths(
        env,
        model,
        path_ids=path_ids,
        deterministic=not args.stochastic,
        total_training_env_steps=int(config["training_config"]["total_timesteps"]),
    )

    out_dir = run_dir / "plots" / f"{args.split}_path_{args.path_id}"
    if args.n_paths > 1:
        out_dir = run_dir / "plots" / f"{args.split}_paths_{args.path_id}_{path_ids[-1]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_prices = split_paths[path_ids]
    _save(plot_price_paths(selected_prices, n_paths=args.n_paths), out_dir / "price.png")
    _save(plot_agent_actions(trajectories, n_paths=args.n_paths), out_dir / "actions.png")
    _save(plot_storage_levels(trajectories, n_paths=args.n_paths), out_dir / "storage.png")
    _save(
        plot_cumulative_cashflows(trajectories, n_paths=args.n_paths),
        out_dir / "cashflow.png",
    )
    _save(plot_price_action_scatter(trajectories), out_dir / "price_action_scatter.png")

    print(f"Saved plots to {out_dir.resolve()}")
    print(metrics)


def _load_model(algorithm_name: str, model_path: Path, env: GasStorageEnv):
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


def _model_path(run_dir: Path, model_name: str) -> Path:
    """Returns the model path for a model selector."""
    if model_name == "best":
        return run_dir / "best_validation_model.zip"
    return run_dir / "final_model.zip"


def _save(figure: plt.Figure, path: Path) -> None:
    """Saves and closes a Matplotlib figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
