"""Gymnasium-compatible gas storage valuation environment."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gas_storage_rl.data.path_dataset import PathDataset
from gas_storage_rl.envs.storage_dynamics import (
    StorageParams,
    clip_storage_action,
    clip_storage_action_to_terminal_feasibility,
)


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
        clipping_variant: str = "v1",
        clip_penalty_factor: float = 1.0,
        initial_inventory_mean_fraction: float | None = None,
        initial_inventory_std_fraction: float = 0.0,
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
        self.clipping_variant = str(clipping_variant).lower()
        if self.clipping_variant not in {"v0", "v1", "v2"}:
            raise ValueError("clipping_variant must be one of: v0, v1, v2")
        self.clip_penalty_factor = float(clip_penalty_factor)
        if self.clip_penalty_factor < 0.0:
            raise ValueError("clip_penalty_factor must be non-negative")
        self.initial_inventory_mean_fraction = initial_inventory_mean_fraction
        self.initial_inventory_std_fraction = float(initial_inventory_std_fraction)
        if (
            self.initial_inventory_mean_fraction is not None
            and not 0.0 <= self.initial_inventory_mean_fraction <= 1.0
        ):
            raise ValueError("initial_inventory_mean_fraction must be in [0, 1]")
        if self.initial_inventory_std_fraction < 0.0:
            raise ValueError("initial_inventory_std_fraction must be non-negative")
        self.fixed_path_id = fixed_path_id
        self.rng = np.random.default_rng(seed)
        self.episode_length = dataset.episode_length
        self.mean_training_price = float(np.mean(dataset.get_paths("train")))
        self.lambda_terminal = self.penalty_factor * self.mean_training_price
        self.lambda_clip = self.clip_penalty_factor * self.mean_training_price

        high = np.array(
            [np.inf, np.inf, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        low = np.array([0.0, -np.inf, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.current_step = 0
        self.current_start_index = 0
        self.initial_inventory = float(storage_params.initial_inventory)
        self.target_terminal_inventory = float(
            storage_params.target_terminal_inventory
        )
        self.storage_level = self.initial_inventory
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
        if "path_id" in options:
            self.current_path_id = int(options["path_id"])
        elif self.fixed_path_id is not None:
            self.current_path_id = int(self.fixed_path_id)
        else:
            self.current_path_id = self.dataset.sample_path_id(self.split, self.rng)
        raw_path = self.dataset.get_raw_path(self.split, self.current_path_id)
        max_start = len(raw_path) - self.episode_length
        if "start_index" in options:
            self.current_start_index = int(options["start_index"])
        elif (
            self.split == "train"
            and "path_id" not in options
            and self.fixed_path_id is None
            and max_start > 0
        ):
            self.current_start_index = int(self.rng.integers(0, max_start + 1))
        else:
            self.current_start_index = int(
                self.dataset.get_start_indices(self.split)[self.current_path_id]
            )
        self.current_path = self.dataset.get_path_window(
            self.split,
            self.current_path_id,
            self.current_start_index,
        )
        if "initial_inventory" in options:
            initial_inventory = float(options["initial_inventory"])
        elif (
            self.split == "train"
            and "path_id" not in options
            and self.fixed_path_id is None
            and self.initial_inventory_mean_fraction is not None
        ):
            inventory_fraction = self.rng.normal(
                self.initial_inventory_mean_fraction,
                self.initial_inventory_std_fraction,
            )
            initial_inventory = float(
                np.clip(inventory_fraction, 0.0, 1.0)
                * self.storage_params.capacity
            )
        else:
            initial_inventory = float(
                self.dataset.get_initial_inventories(
                    self.split,
                    default=self.storage_params.initial_inventory,
                )[self.current_path_id]
            )
        self.initial_inventory = initial_inventory
        self.target_terminal_inventory = initial_inventory
        self.storage_level = initial_inventory
        return self._observation(), {
            "path_id": self.current_path_id,
            "split": self.split,
            "current_step": self.current_step,
            "start_index": self.current_start_index,
            "initial_inventory": self.initial_inventory,
            "target_terminal_inventory": self.target_terminal_inventory,
        }

    def step(
        self,
        action: np.ndarray | list[float] | float,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Applies a storage action for one decision day."""
        requested_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        requested_action = float(np.clip(requested_action, -1.0, 1.0))
        price = float(self.current_path[self.current_step])
        rate_capacity_clipped_action = clip_storage_action(
            requested_action=requested_action,
            storage_level=self.storage_level,
            params=self.storage_params,
        )
        if self.clipping_variant == "v0":
            executed_action = rate_capacity_clipped_action
        else:
            remaining_steps_after_action = self.episode_length - self.current_step - 1
            executed_action = clip_storage_action_to_terminal_feasibility(
                requested_action=requested_action,
                storage_level=self.storage_level,
                params=self.storage_params,
                remaining_steps_after_action=remaining_steps_after_action,
                target_inventory=self.target_terminal_inventory,
            )
        terminal_feasibility_clipped = (
            abs(executed_action - rate_capacity_clipped_action) > 1e-8
        )
        terminal_clip_distance = abs(
            rate_capacity_clipped_action - executed_action
        )
        clip_penalty = (
            -self.lambda_clip * terminal_clip_distance
            if self.clipping_variant == "v2"
            else 0.0
        )
        previous_level = self.storage_level
        self.storage_level += executed_action
        raw_cashflow = -executed_action * price

        is_final = self.current_step == self.episode_length - 1
        terminal_penalty = 0.0
        if is_final:
            deviation = abs(self.storage_level - self.target_terminal_inventory)
            terminal_penalty = -self.lambda_terminal * deviation

        raw_reward = raw_cashflow + terminal_penalty
        shaped_raw_reward = raw_reward + clip_penalty
        economic_scaled_reward = raw_reward / self.reward_scale
        shaped_scaled_reward = shaped_raw_reward / self.reward_scale
        info = {
            "clipping_variant": self.clipping_variant,
            "requested_action": requested_action,
            "rate_capacity_clipped_action": rate_capacity_clipped_action,
            "executed_action": executed_action,
            "terminal_feasibility_clipped": terminal_feasibility_clipped,
            "terminal_clip_distance": terminal_clip_distance,
            "clip_penalty": clip_penalty,
            "storage_level": self.storage_level,
            "previous_storage_level": previous_level,
            "initial_inventory": self.initial_inventory,
            "target_terminal_inventory": self.target_terminal_inventory,
            "price": price,
            "reward_scale": self.reward_scale,
            "raw_cashflow": raw_cashflow,
            "terminal_penalty": terminal_penalty,
            "raw_reward": raw_reward,
            "shaped_raw_reward": shaped_raw_reward,
            "economic_scaled_reward": economic_scaled_reward,
            "shaped_scaled_reward": shaped_scaled_reward,
            "scaled_reward": shaped_scaled_reward,
            "current_step": self.current_step,
            "start_index": self.current_start_index,
            "path_id": self.current_path_id,
            "split": self.split,
        }
        self.current_step += 1
        terminated = is_final
        truncated = False
        return (
            self._observation(),
            float(shaped_scaled_reward),
            terminated,
            truncated,
            info,
        )

    def _observation(self) -> np.ndarray:
        """Builds a normalized observation."""
        step_index = min(self.current_step, self.episode_length - 1)
        price = float(self.current_path[step_index])
        current_date = self._current_date(step_index)
        days_in_year = 366 if _is_leap_year(current_date.year) else 365
        day_angle = (
            2.0 * np.pi * (current_date.timetuple().tm_yday - 1) / days_in_year
        )
        time_denominator = max(self.episode_length - 1, 1)
        remaining_time = (self.episode_length - 1 - step_index) / time_denominator
        return np.array(
            [
                self.storage_level / self.storage_params.capacity,
                price / self.price_scale,
                np.sin(day_angle),
                np.cos(day_angle),
                remaining_time,
                self.target_terminal_inventory / self.storage_params.capacity,
            ],
            dtype=np.float32,
        )

    def _current_date(self, step_index: int) -> date:
        """Returns the calendar date represented by an episode step."""
        if (
            self.dataset.base_dates_by_split is not None
            and self.split in self.dataset.base_dates_by_split
        ):
            base_date = datetime.strptime(
                self.dataset.base_dates_by_split[self.split], "%Y-%m-%d"
            ).date()
            return base_date + timedelta(
                days=self.current_start_index + step_index
            )
        if self.dataset.date_ranges_by_split is None:
            return date(2001, 1, 1) + timedelta(days=step_index)
        date_ranges = self.dataset.date_ranges_by_split.get(self.split)
        if not date_ranges:
            return date(2001, 1, 1) + timedelta(days=step_index)
        start_date = datetime.strptime(
            date_ranges[self.current_path_id]["start_date"], "%Y-%m-%d"
        ).date()
        return start_date + timedelta(days=step_index)


def _is_leap_year(year: int) -> bool:
    """Returns whether a Gregorian calendar year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
