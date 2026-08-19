"""Manual holdout evaluation for test and historical backtest splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.backtest import build_backtest_evaluation_dataset
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.logging.progress import CliProgress
from gas_storage_rl.training.config import build_environment, load_config


def main() -> None:
    """Runs manual evaluation on test or historical backtest data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", required=True, choices=["validation", "test", "backtest"])
    parser.add_argument(
        "--write-final-episode-metrics",
        action="store_true",
        help="Write final_episode_metrics_<split>.csv for per-path comparisons.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    progress = CliProgress("run_holdout_evaluation", total=5)
    progress.step(1, "loading config")
    config = load_config(run_dir / "config.json")
    algorithm_name = config["agent_config"]["algorithm_name"]
    progress.step(2, "building environment")
    dataset, storage_params, env_kwargs = build_environment(config)

    if args.split == "backtest":
        progress.step(3, "building backtest dataset")
        dataset = build_backtest_evaluation_dataset(config, dataset)
    else:
        progress.step(3, f"using synthetic {args.split} split")

    env = GasStorageEnv(dataset, args.split, storage_params, **env_kwargs)
    progress.step(4, "loading model")
    model = _load_model(algorithm_name, run_dir / "final_model", env)
    total_training_env_steps = int(config["training_config"]["total_timesteps"])
    progress.step(5, f"evaluating split={args.split}")
    metrics, trajectories = evaluate_policy_on_paths(
        env,
        model,
        deterministic=bool(config["evaluation_config"].get("deterministic", True)),
        total_training_env_steps=total_training_env_steps,
    )
    metrics["algorithm_name"] = algorithm_name
    metrics["holdout_split"] = args.split

    output_name = f"holdout_{args.split}_metrics.json"
    with (run_dir / output_name).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    if args.write_final_episode_metrics:
        _write_csv(
            run_dir / f"final_episode_metrics_{args.split}.csv",
            _episode_rows(
                trajectories,
                method=algorithm_name,
                split=args.split,
                dataset=dataset,
                seed=config["seeds"].get("agent_seed"),
            ),
        )
    progress.finish("done")
    print(json.dumps({"output": str(run_dir / output_name), **metrics}, indent=2))


def _load_model(algorithm_name: str, model_path: Path, env: GasStorageEnv) -> Any:
    """Loads a Stable-Baselines3 model for the requested algorithm.
    
    Args:
        algorithm_name: Algorithm name value.
        model_path: Model path value.
        env: Environment used for rollout or training.
    
    Returns:
        Load model result.
    
    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    if algorithm_name == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(model_path, env=env)
    if algorithm_name == "sac":
        from stable_baselines3 import SAC

        return SAC.load(model_path, env=env)
    if algorithm_name == "td3":
        from stable_baselines3 import TD3

        return TD3.load(model_path, env=env)
    raise ValueError(f"Unsupported algorithm_name: {algorithm_name}")


def _episode_rows(
    trajectories: list[dict[str, Any]],
    method: str,
    split: str,
    dataset: PathDataset,
    seed: int | None,
) -> list[dict[str, Any]]:
    """Returns one final metrics row per evaluated RL episode.
    
    Args:
        trajectories: Rollout trajectories to summarize or transform.
        method: Method value.
        split: Dataset split name.
        dataset: Path dataset used by the environment or evaluator.
        seed: Random seed for deterministic behavior.
    
    Returns:
        Episode rows result.

    """
    from gas_storage_rl.evaluation.metrics import summarize_episode_infos

    date_ranges = (
        None
        if dataset.date_ranges_by_split is None
        else dataset.date_ranges_by_split.get(split)
    )
    rows = []
    for trajectory in trajectories:
        infos = trajectory["infos"]
        if not infos:
            continue
        path_id = int(trajectory["path_id"])
        summary = summarize_episode_infos(infos)
        rows.append(
            {
                "method": method,
                "algorithm_name": method,
                "method_type": "rl",
                "split": split,
                "path_id": path_id,
                "start_index": int(infos[0]["start_index"]),
                "start_date": (
                    None if date_ranges is None else date_ranges[path_id].get("start_date")
                ),
                "initial_inventory": float(infos[0]["initial_inventory"]),
                "target_inventory": float(infos[-1]["target_terminal_inventory"]),
                "seed": seed,
                **summary,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes rows in CSV format.
    
    Args:
        path: Filesystem path to read from or write to.
        rows: Rows to read, transform, or persist.

    """
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
