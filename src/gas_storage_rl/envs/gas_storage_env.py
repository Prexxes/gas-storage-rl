"""Gymnasium-compatible gas storage valuation environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.storage_dynamics import StorageParams, clip_storage_action


class GasStorageEnv(gym.Env):
    """Continuous-control gas storage environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: PathDataset,
        split: str,
        storage_params: StorageParams,
        price_scale: float = 50.0,
        reward_scale: float | None = None,
        penalty_factor: float = 0.5,
        fixed_path_id: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Initializes the environment."""
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.storage_params = storage_params
        self.price_scale = float(price_scale)
        self.reward_scale = float(reward_scale or price_scale)
        self.penalty_factor = float(penalty_factor)
        self.fixed_path_id = fixed_path_id
        self.rng = np.random.default_rng(seed)
        self.episode_length = dataset.episode_length
        self.mean_training_price = float(np.mean(dataset.get_paths("train")))
        self.lambda_terminal = self.penalty_factor * self.mean_training_price

        high = np.array([np.inf, np.inf, 1.0], dtype=np.float32)
        low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.current_step = 0
        self.storage_level = storage_params.initial_inventory
        self.current_path_id = 0
        self.current_path = self.dataset.get_path(split, 0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the environment and returns the first observation."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.current_step = 0
        self.storage_level = float(self.storage_params.initial_inventory)
        if "path_id" in options:
            self.current_path_id = int(options["path_id"])
        elif self.fixed_path_id is not None:
            self.current_path_id = int(self.fixed_path_id)
        else:
            self.current_path_id = self.dataset.sample_path_id(self.split, self.rng)
        self.current_path = self.dataset.get_path(self.split, self.current_path_id)
        return self._observation(), {
            "path_id": self.current_path_id,
            "split": self.split,
            "current_step": self.current_step,
        }

    def step(
        self,
        action: np.ndarray | list[float] | float,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Applies a storage action for one decision day."""
        requested_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        requested_action = float(np.clip(requested_action, -1.0, 1.0))
        price = float(self.current_path[self.current_step])
        executed_action = clip_storage_action(
            requested_action=requested_action,
            storage_level=self.storage_level,
            params=self.storage_params,
        )
        previous_level = self.storage_level
        self.storage_level += executed_action
        raw_cashflow = -executed_action * price

        is_final = self.current_step == self.episode_length - 1
        terminal_penalty = 0.0
        if is_final:
            deviation = abs(
                self.storage_level - self.storage_params.target_terminal_inventory
            )
            terminal_penalty = -self.lambda_terminal * deviation

        raw_reward = raw_cashflow + terminal_penalty
        scaled_reward = raw_reward / self.reward_scale
        info = {
            "requested_action": requested_action,
            "executed_action": executed_action,
            "storage_level": self.storage_level,
            "previous_storage_level": previous_level,
            "price": price,
            "raw_cashflow": raw_cashflow,
            "terminal_penalty": terminal_penalty,
            "raw_reward": raw_reward,
            "scaled_reward": scaled_reward,
            "current_step": self.current_step,
            "path_id": self.current_path_id,
            "split": self.split,
        }
        self.current_step += 1
        terminated = is_final
        truncated = False
        return self._observation(), float(scaled_reward), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        """Builds a normalized observation."""
        step_index = min(self.current_step, self.episode_length - 1)
        price = float(self.current_path[step_index])
        return np.array(
            [
                self.storage_level / self.storage_params.capacity,
                price / self.price_scale,
                step_index / (self.episode_length - 1),
            ],
            dtype=np.float32,
        )
