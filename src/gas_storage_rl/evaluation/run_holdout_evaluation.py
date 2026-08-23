"""Manual holdout evaluation for test and historical backtest splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from gas_storage_rl.baselines.perfect_foresight import PerfectForesightBaseline
from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.backtest import build_backtest_evaluation_dataset
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.evaluation.metrics import interquartile_mean
from gas_storage_rl.logging.progress import CliProgress
from gas_storage_rl.plotting.statistics import bootstrap_percentile_statistic_ci
from gas_storage_rl.training.config import build_environment, load_config

MODEL_CHECKPOINTS = {
    "best_validation": "best_validation_model",
    "best_risk_adjusted_validation": "best_risk_adjusted_validation_model",
    "final": "final_model",
}


def main() -> None:
    """Runs manual evaluation on test or historical backtest data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=["validation", "test", "backtest"],
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHECKPOINTS),
        default="best_validation",
        help="Model checkpoint to evaluate. Defaults to best_validation_model.zip.",
    )
    parser.add_argument(
        "--write-final-episode-metrics",
        action="store_true",
        help="Write final_episode_metrics_<split>.csv for per-path comparisons.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    progress = CliProgress("run_holdout_evaluation", total=5)
    progress.step(1, "loading config")
    run_dir = Path(args.run_dir)
    progress.step(2, "building environment")
    progress.step(3, f"using split={args.split}")
    progress.step(4, f"loading model={args.model}")
    progress.step(5, f"evaluating split={args.split}")
    metrics = evaluate_holdout_run(
        run_dir,
        args.split,
        model_checkpoint=args.model,
        write_final_episode_metrics=args.write_final_episode_metrics,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    progress.finish("done")
    print(
        json.dumps(
            {"output": str(run_dir / f"holdout_{args.split}_metrics.json"), **metrics},
            indent=2,
        )
    )


def evaluate_holdout_run(
    run_dir: str | Path,
    split: str,
    *,
    model_checkpoint: str = "best_validation",
    write_final_episode_metrics: bool = False,
    perfect_foresight_references: list[dict[str, Any]] | None = None,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluates one trained run on a holdout split and writes run artifacts.

    Args:
        run_dir: Experiment run directory containing ``config.json`` and models.
        split: Dataset split to evaluate.
        model_checkpoint: Named model checkpoint to load.
        write_final_episode_metrics: Whether to write per-episode policy metrics.
        perfect_foresight_references: Optional precomputed path-wise references.
        n_bootstrap: Number of percentile bootstrap resamples for IQM CI.
        bootstrap_seed: Seed used for deterministic bootstrap intervals.

    Returns:
        Holdout metric dictionary.
    """
    run_path = Path(run_dir)
    config = load_config(run_path / "config.json")
    algorithm_name = config["agent_config"]["algorithm_name"]
    dataset, storage_params, env_kwargs = build_environment(config)

    if split == "backtest":
        dataset = build_backtest_evaluation_dataset(config, dataset)

    env = GasStorageEnv(dataset, split, storage_params, **env_kwargs)
    model_path = _model_path(run_path, model_checkpoint)
    model = _load_model(
        algorithm_name,
        model_path,
        env,
        device=_model_device(config, algorithm_name),
    )
    total_training_env_steps = int(config["training_config"]["total_timesteps"])
    metrics, trajectories = evaluate_policy_on_paths(
        env,
        model,
        deterministic=bool(
            config.get("evaluation_config", {}).get("deterministic", True)
        ),
        total_training_env_steps=total_training_env_steps,
    )
    if perfect_foresight_references is None:
        perfect_foresight_references = perfect_foresight_reference_rows(
            dataset,
            split,
            storage_params,
            env.lambda_terminal,
        )
    episode_rows = _episode_rows(
        trajectories,
        method=algorithm_name,
        split=split,
        dataset=dataset,
        seed=config["seeds"].get("agent_seed"),
        perfect_foresight_references=perfect_foresight_references,
    )
    _add_holdout_reference_metrics(
        metrics,
        episode_rows,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    metrics["algorithm_name"] = algorithm_name
    metrics["holdout_split"] = split
    metrics["model_checkpoint"] = model_checkpoint
    metrics["model_path"] = str(model_path.with_suffix(".zip"))
    metrics["n_episodes"] = len(episode_rows)
    metrics["iqm_bootstrap_n"] = int(n_bootstrap)
    metrics["iqm_bootstrap_confidence_level"] = 0.95

    output_name = f"holdout_{split}_metrics.json"
    _write_json(run_path / output_name, metrics)
    if write_final_episode_metrics:
        _write_csv(
            run_path / f"final_episode_metrics_{split}.csv",
            episode_rows,
        )
    return metrics


def perfect_foresight_reference_rows(
    dataset: PathDataset,
    split: str,
    storage_params: Any,
    lambda_terminal: float,
) -> list[dict[str, Any]]:
    """Solves perfect-foresight references for all paths in a split.

    Args:
        dataset: Dataset containing the evaluated split.
        split: Dataset split name.
        storage_params: Storage parameter object.
        lambda_terminal: Terminal inventory penalty coefficient.

    Returns:
        One reference row per path.
    """
    perfect_foresight = PerfectForesightBaseline(storage_params, lambda_terminal)
    paths = dataset.get_paths(split)
    initial_inventories = dataset.get_initial_inventories(
        split,
        storage_params.initial_inventory,
    )
    date_ranges = (
        None
        if dataset.date_ranges_by_split is None
        else dataset.date_ranges_by_split.get(split)
    )
    trajectories = perfect_foresight.solve_paths(
        paths,
        split,
        date_ranges,
        initial_inventories,
        initial_inventories,
    )
    rows = []
    for trajectory in trajectories:
        rows.append(
            {
                "split": split,
                "path_id": int(trajectory["path_id"]),
                "start_date": trajectory.get("start_date"),
                "end_date": trajectory.get("end_date"),
                "initial_inventory": float(trajectory["initial_inventory"]),
                "target_inventory": float(trajectory["target_inventory"]),
                "episode_perfect_foresight_return_raw": float(
                    trajectory["objective_value"]
                ),
                "perfect_foresight_terminal_deviation": float(
                    trajectory["terminal_deviation"]
                ),
                "perfect_foresight_success": bool(trajectory["success"]),
                "perfect_foresight_actions": trajectory["actions"],
                "perfect_foresight_storage_levels": trajectory["storage_levels"],
            }
        )
    return rows


def _model_path(run_dir: Path, model_checkpoint: str) -> Path:
    """Returns the filesystem path stem for a named model checkpoint.

    Args:
        run_dir: Experiment run directory.
        model_checkpoint: Named checkpoint.

    Returns:
        Model path stem accepted by Stable-Baselines3 loaders.

    Raises:
        ValueError: If the checkpoint name is unknown.
        FileNotFoundError: If the checkpoint zip file is absent.
    """
    if model_checkpoint not in MODEL_CHECKPOINTS:
        raise ValueError(f"Unsupported model checkpoint: {model_checkpoint}")
    path = run_dir / MODEL_CHECKPOINTS[model_checkpoint]
    if not path.with_suffix(".zip").exists() and not path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {path.with_suffix('.zip')}")
    return path


def _model_device(config: dict[str, Any], algorithm_name: str) -> str:
    """Returns the configured SB3 device for original or effective run configs."""
    agent_config = config.get("agent_config", {})
    candidates = (
        agent_config.get(algorithm_name, {}),
        agent_config.get("hyperparameters", {}),
        agent_config.get("effective_hyperparameters", {}),
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("device") is not None:
            return str(candidate["device"])
    return "auto"


def _load_model(
    algorithm_name: str,
    model_path: Path,
    env: GasStorageEnv,
    device: str = "cpu",
) -> Any:
    """Loads a Stable-Baselines3 model for the requested algorithm.
    
    Args:
        algorithm_name: Algorithm name value.
        model_path: Model path value.
        env: Environment used for rollout or training.
        device: Torch device used when loading the model.
    
    Returns:
        Load model result.
    
    Raises:
        ValueError: If an input value or configuration is invalid.

    """
    if algorithm_name == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(model_path, env=env, device=device)
    if algorithm_name == "sac":
        from stable_baselines3 import SAC

        return SAC.load(model_path, env=env, device=device)
    if algorithm_name == "td3":
        from stable_baselines3 import TD3

        return TD3.load(model_path, env=env, device=device)
    raise ValueError(f"Unsupported algorithm_name: {algorithm_name}")


def _episode_rows(
    trajectories: list[dict[str, Any]],
    method: str,
    split: str,
    dataset: PathDataset,
    seed: int | None,
    perfect_foresight_references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Returns one final metrics row per evaluated RL episode.
    
    Args:
        trajectories: Rollout trajectories to summarize or transform.
        method: Method value.
        split: Dataset split name.
        dataset: Path dataset used by the environment or evaluator.
        seed: Random seed for deterministic behavior.
        perfect_foresight_references: Perfect-foresight rows keyed by path id.
    
    Returns:
        Episode rows result.

    """
    from gas_storage_rl.evaluation.metrics import summarize_episode_infos

    date_ranges = (
        None
        if dataset.date_ranges_by_split is None
        else dataset.date_ranges_by_split.get(split)
    )
    references_by_path_id = {
        int(row["path_id"]): row for row in perfect_foresight_references
    }
    rows = []
    for trajectory in trajectories:
        infos = trajectory["infos"]
        if not infos:
            continue
        path_id = int(trajectory["path_id"])
        summary = summarize_episode_infos(infos)
        reference = references_by_path_id[path_id]
        perfect_foresight_return = float(
            reference["episode_perfect_foresight_return_raw"]
        )
        summary.update(
            _relative_perfect_foresight_metrics(
                float(summary["episode_return_raw"]),
                perfect_foresight_return,
            )
        )
        rows.append(
            {
                "method": method,
                "algorithm_name": method,
                "method_type": "rl",
                "split": split,
                "path_id": path_id,
                "start_index": int(infos[0]["start_index"]),
                "start_date": (
                    None
                    if date_ranges is None
                    else date_ranges[path_id].get("start_date")
                ),
                "initial_inventory": float(infos[0]["initial_inventory"]),
                "target_inventory": float(infos[-1]["target_terminal_inventory"]),
                "seed": seed,
                "episode_perfect_foresight_return_raw": perfect_foresight_return,
                **summary,
            }
        )
    return rows


def _relative_perfect_foresight_metrics(
    episode_return: float,
    perfect_foresight_return: float,
) -> dict[str, float]:
    """Computes path-wise ratio and gap to a perfect-foresight return."""
    if abs(perfect_foresight_return) <= 1e-12:
        return {
            "episode_perfect_foresight_ratio": float("nan"),
            "episode_optimality_gap": float("nan"),
        }
    return {
        "episode_perfect_foresight_ratio": float(
            episode_return / perfect_foresight_return
        ),
        "episode_optimality_gap": float(
            (perfect_foresight_return - episode_return)
            / abs(perfect_foresight_return)
        ),
    }


def _add_holdout_reference_metrics(
    metrics: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> None:
    """Adds IQM bootstrap and perfect-foresight comparison metrics."""
    returns = np.asarray(
        [row["episode_return_raw"] for row in episode_rows],
        dtype=np.float64,
    )
    ci_lower, ci_upper = bootstrap_percentile_statistic_ci(
        returns,
        statistic=interquartile_mean,
        n_bootstrap=n_bootstrap,
        random_seed=bootstrap_seed,
    )
    ratios = _finite_values(
        [row["episode_perfect_foresight_ratio"] for row in episode_rows]
    )
    gaps = _finite_values([row["episode_optimality_gap"] for row in episode_rows])
    references = _finite_values(
        [row["episode_perfect_foresight_return_raw"] for row in episode_rows]
    )
    metrics["interquartile_mean_return_raw_ci_lower"] = ci_lower
    metrics["interquartile_mean_return_raw_ci_upper"] = ci_upper
    metrics["mean_perfect_foresight_return_raw"] = _nanmean(references)
    metrics["mean_perfect_foresight_ratio"] = _nanmean(ratios)
    metrics["mean_optimality_gap"] = _nanmean(gaps)


def _finite_values(values: list[Any]) -> np.ndarray:
    """Returns finite float values from a possibly sparse sequence."""
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _nanmean(values: np.ndarray) -> float:
    """Returns NaN for empty arrays instead of raising a warning."""
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes a JSON document."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


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
