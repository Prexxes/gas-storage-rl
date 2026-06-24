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


def load_trajectory_rows(trajectory_path: str | Path) -> list[dict[str, Any]]:
    """Loads serialized perfect-foresight trajectories from a JSONL file."""
    trajectories = []
    with Path(trajectory_path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                trajectories.append(json.loads(line))
    if not trajectories:
        raise ValueError("No trajectory samples found")
    return trajectories


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


def discounted_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Calculates discounted returns-to-go for a sequence of transitions."""
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = (
            float(rewards[index])
            + float(gamma) * (1.0 - float(dones[index])) * running_return
        )
        returns[index] = running_return
    return returns


def trajectory_rows_to_transitions(
    trajectories: list[dict[str, Any]],
    dataset: Any,
    storage_params: Any,
    env_kwargs: dict[str, Any],
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolls expert actions through environments and returns training targets.

    The environment, rather than a duplicate reward implementation, defines the
    reward and executed action. This keeps critic targets identical to the ones
    used during subsequent RL training.
    """
    environments: dict[str, GasStorageEnv] = {}
    observations = []
    requested_actions = []
    executed_actions = []
    returns = []

    for trajectory in trajectories:
        split = str(trajectory.get("split", "pretrain"))
        if split not in dataset.paths_by_split:
            raise ValueError(f"Unknown trajectory split: {split}")
        env = environments.setdefault(
            split,
            GasStorageEnv(dataset, split, storage_params, **env_kwargs),
        )
        reset_options = {
            "path_id": int(trajectory["path_id"]),
            "initial_inventory": float(
                trajectory.get("initial_inventory", storage_params.initial_inventory)
            ),
        }
        observation, _ = env.reset(options=reset_options)
        path_observations = []
        path_requested_actions = []
        path_executed_actions = []
        path_rewards = []
        path_dones = []

        for expert_action in trajectory["actions"]:
            requested_action = np.asarray([expert_action], dtype=np.float32)
            next_observation, reward, terminated, truncated, info = env.step(
                requested_action
            )
            path_observations.append(observation)
            path_requested_actions.append(requested_action)
            path_executed_actions.append(
                np.asarray([info["executed_action"]], dtype=np.float32)
            )
            path_rewards.append(float(reward))
            path_dones.append(float(terminated or truncated))
            observation = next_observation
            if terminated or truncated:
                break

        path_rewards_array = np.asarray(path_rewards, dtype=np.float32)
        path_dones_array = np.asarray(path_dones, dtype=np.float32)
        observations.extend(path_observations)
        requested_actions.extend(path_requested_actions)
        executed_actions.extend(path_executed_actions)
        returns.extend(discounted_returns(path_rewards_array, path_dones_array, gamma))

    if not observations:
        raise ValueError("No trajectory transitions found")
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(requested_actions, dtype=np.float32),
        np.asarray(executed_actions, dtype=np.float32),
        np.asarray(returns, dtype=np.float32),
    )


def train_behavior_cloning(
    model: Any,
    algorithm_name: str,
    observations: np.ndarray,
    requested_actions: np.ndarray,
    executed_actions: np.ndarray,
    returns: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    value_loss_coefficient: float = 0.5,
    progress: CliProgress | None = None,
) -> list[dict[str, float | int]]:
    """Pretrains actor and critic networks from expert transitions."""
    torch.manual_seed(seed)
    device = model.device
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    requested_action_tensor = torch.as_tensor(
        requested_actions, dtype=torch.float32, device=device
    )
    executed_action_tensor = torch.as_tensor(
        executed_actions, dtype=torch.float32, device=device
    )
    return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    dataset = TensorDataset(
        obs_tensor,
        requested_action_tensor,
        executed_action_tensor,
        return_tensor,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    ppo_optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    actor_optimizer = None
    critic_optimizer = None
    if algorithm_name in {"sac", "td3"}:
        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=learning_rate)
        critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        actor_losses = []
        critic_losses = []
        total_losses = []
        for batch_obs, batch_requested_actions, batch_executed_actions, batch_returns in loader:
            predicted_actions = predict_actor_actions(model, batch_obs, algorithm_name)
            actor_loss = torch.nn.functional.mse_loss(
                predicted_actions, batch_requested_actions
            )
            if algorithm_name == "ppo":
                predicted_values = model.policy.predict_values(batch_obs).flatten()
                critic_loss = torch.nn.functional.mse_loss(
                    predicted_values, batch_returns
                )
                total_loss = actor_loss + value_loss_coefficient * critic_loss
                ppo_optimizer.zero_grad()
                total_loss.backward()
                ppo_optimizer.step()
            else:
                q1_values, q2_values = model.critic(
                    batch_obs, batch_executed_actions
                )
                critic_loss = torch.nn.functional.mse_loss(
                    q1_values.flatten(), batch_returns
                ) + torch.nn.functional.mse_loss(q2_values.flatten(), batch_returns)
                assert actor_optimizer is not None
                assert critic_optimizer is not None
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                total_loss = actor_loss + critic_loss
            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            total_losses.append(float(total_loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(total_losses)),
                "actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses)),
                "n_samples": int(len(dataset)),
            }
        )
        if progress is not None:
            progress.update(epoch, f"loss={history[-1]['loss']:.6f}")
    if algorithm_name in {"sac", "td3"}:
        model.critic_target.load_state_dict(model.critic.state_dict())
        if algorithm_name == "td3":
            model.actor_target.load_state_dict(model.actor.state_dict())
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
    progress.step(3, "loading expert transitions")
    trajectories = load_trajectory_rows(trajectory_path)
    gamma = float(getattr(model, "gamma", 0.99))
    (
        observations,
        requested_actions,
        executed_actions,
        returns,
    ) = trajectory_rows_to_transitions(
        trajectories,
        dataset,
        storage_params,
        env_kwargs,
        gamma,
    )
    progress.step(4, f"training {epochs} epochs on {len(observations)} samples")
    history = train_behavior_cloning(
        model,
        algorithm_name,
        observations,
        requested_actions,
        executed_actions,
        returns,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=int(config["seeds"]["agent_seed"]),
        value_loss_coefficient=float(
            config.get("pretraining_config", {}).get("value_loss_coefficient", 0.5)
        ),
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
        "gamma": gamma,
        "value_loss_coefficient": float(
            config.get("pretraining_config", {}).get("value_loss_coefficient", 0.5)
        ),
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
