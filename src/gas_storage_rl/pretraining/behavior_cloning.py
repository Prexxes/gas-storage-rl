"""Behavior-cloning pretraining from perfect-foresight trajectories."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gas_storage_rl.agents.sb3_factory import make_sb3_agent
from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.logging.progress import CliProgress
from gas_storage_rl.training.config import build_environment, load_config


def _json_default(value: Any) -> str:
    """JSON fallback serializer."""
    return str(value)


def config_hash(config: dict[str, Any]) -> str:
    """Returns a short stable hash for a loaded configuration."""
    serialized = json.dumps(config, sort_keys=True, default=_json_default)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]


def create_pretraining_run_dir(
    base_dir: str | Path,
    config_name: str,
    algorithm_name: str,
    hash_value: str,
) -> tuple[Path, str]:
    """Creates a human-readable pretraining run directory."""
    safe_config_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in config_name
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{safe_config_name}-{algorithm_name}-{hash_value}"
    run_dir = Path(base_dir) / "pretraining" / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{timestamp}-{safe_config_name}-{algorithm_name}-{hash_value}-{suffix}"
        run_dir = Path(base_dir) / "pretraining" / run_id
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def load_trajectory_samples(
    trajectory_path: str | Path,
    capacity: float,
    price_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds observation/action pairs from perfect-foresight JSONL trajectories."""
    trajectories = []
    with Path(trajectory_path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectory_rows_to_samples(
        trajectories,
        capacity=capacity,
        price_scale=price_scale,
    )


def trajectory_rows_to_samples(
    trajectories: list[dict[str, Any]],
    capacity: float,
    price_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds observation/action pairs from perfect-foresight trajectory rows."""
    observations = []
    actions = []
    for trajectory in trajectories:
        prices = np.asarray(trajectory["prices"], dtype=np.float32)
        storage_levels = np.asarray(trajectory["storage_levels"], dtype=np.float32)
        path_actions = np.asarray(trajectory["actions"], dtype=np.float32)
        target_inventory = float(trajectory.get("target_inventory", storage_levels[0]))
        horizon = len(path_actions)
        denominator = max(horizon - 1, 1)
        start_date_value = trajectory.get("start_date")
        start_date = (
            datetime.strptime(start_date_value, "%Y-%m-%d").date()
            if start_date_value
            else date(2001, 1, 1)
        )
        for step in range(horizon):
            current_date = start_date + timedelta(days=step)
            days_in_year = 366 if _is_leap_year(current_date.year) else 365
            day_angle = (
                2.0 * np.pi * (current_date.timetuple().tm_yday - 1) / days_in_year
            )
            observations.append(
                [
                    float(storage_levels[step] / capacity),
                    float(prices[step] / price_scale),
                    float(np.sin(day_angle)),
                    float(np.cos(day_angle)),
                    float((horizon - 1 - step) / denominator),
                    float(target_inventory / capacity),
                ]
            )
            actions.append([float(path_actions[step])])
    if not observations:
        raise ValueError("No trajectory samples found")
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def _is_leap_year(year: int) -> bool:
    """Returns whether a Gregorian calendar year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def predict_actor_actions(
    model: Any,
    observations: torch.Tensor,
    algorithm_name: str,
) -> torch.Tensor:
    """Returns differentiable deterministic actor actions."""
    if algorithm_name == "ppo":
        return model.policy._predict(observations, deterministic=True)
    if algorithm_name in {"sac", "td3"}:
        return model.actor(observations)
    raise ValueError(f"Unsupported algorithm_name: {algorithm_name}")


def train_behavior_cloning(
    model: Any,
    algorithm_name: str,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    progress: CliProgress | None = None,
) -> list[dict[str, float | int]]:
    """Trains the model actor to imitate perfect-foresight actions."""
    torch.manual_seed(seed)
    device = model.device
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    dataset = TensorDataset(obs_tensor, action_tensor)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        for batch_obs, batch_actions in loader:
            predicted = predict_actor_actions(model, batch_obs, algorithm_name)
            loss = torch.nn.functional.mse_loss(predicted, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "n_samples": int(len(dataset)),
            }
        )
        if progress is not None:
            progress.update(epoch, f"loss={history[-1]['loss']:.6f}")
    return history


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Writes a JSON payload."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=_json_default)


def run_pretraining(
    config: dict[str, Any],
    config_name: str,
    algorithm_name: str,
    trajectory_path: str | Path,
    output_dir: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Runs behavior-cloning pretraining and persists artifacts."""
    progress = CliProgress("behavior_cloning", total=5, enabled=show_progress)
    config["agent_config"]["algorithm_name"] = algorithm_name
    progress.step(1, "building environment")
    dataset, storage_params, env_kwargs = build_environment(config)
    env = GasStorageEnv(dataset, "train", storage_params, **env_kwargs)
    progress.step(2, "creating agent")
    model = make_sb3_agent(
        algorithm_name,
        env,
        config["agent_config"].get(algorithm_name, {}),
        seed=config["seeds"]["agent_seed"],
    )
    progress.step(3, "loading trajectory samples")
    observations, actions = load_trajectory_samples(
        trajectory_path,
        capacity=storage_params.capacity,
        price_scale=float(env_kwargs["price_scale"]),
    )
    progress.step(4, f"training {epochs} epochs on {len(observations)} samples")
    history = train_behavior_cloning(
        model,
        algorithm_name,
        observations,
        actions,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=int(config["seeds"]["agent_seed"]),
        progress=CliProgress("behavior_cloning_epochs", total=epochs, enabled=show_progress),
    )

    progress.step(5, "writing artifacts")
    hash_value = config_hash(config)
    base_dir = output_dir or config["logging_config"]["run_dir"]
    run_dir, run_id = create_pretraining_run_dir(
        base_dir,
        config_name,
        algorithm_name,
        hash_value,
    )
    model.save(run_dir / "pretrained_model")
    torch.save(model.policy.state_dict(), run_dir / "policy_state_dict.pt")
    write_json(run_dir / "config.json", config)
    metadata = {
        "run_id": run_id,
        "config_name": config_name,
        "config_hash": hash_value,
        "algorithm_name": algorithm_name,
        "trajectory_path": str(trajectory_path),
        "n_samples": int(len(observations)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "python_version": sys.version,
        "platform": platform.platform(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "training_history.json", history)
    progress.finish("done")
    return {
        "run_dir": str(run_dir),
        "policy_state_dict": str(run_dir / "policy_state_dict.pt"),
        "pretrained_model": str(run_dir / "pretrained_model.zip"),
        "final_loss": history[-1]["loss"],
        **metadata,
    }


def main() -> None:
    """Runs behavior-cloning pretraining from perfect-foresight trajectories."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = Path(args.config).stem
    summary = run_pretraining(
        config,
        config_name,
        args.algorithm,
        args.trajectories,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        show_progress=True,
    )
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
