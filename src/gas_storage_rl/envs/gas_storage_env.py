"""Gymnasium-compatible gas storage valuation environment."""

from __future__ import annotations

from datetime import date, datetime, timedelta
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
        feasibility_penalty_factor: float = 0.5,
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
        self.feasibility_penalty_factor = float(feasibility_penalty_factor)
        self.fixed_path_id = fixed_path_id
        self.rng = np.random.default_rng(seed)
        self.episode_length = dataset.episode_length
        self.mean_training_price = float(np.mean(dataset.get_paths("train")))
        self.lambda_terminal = self.penalty_factor * self.mean_training_price
        self.lambda_feasibility = (
            self.feasibility_penalty_factor * self.mean_training_price
        )

        high = np.array([np.inf, np.inf, 1.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.current_step = 0
        self.current_start_index = 0
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
        return self._observation(), {
            "path_id": self.current_path_id,
            "split": self.split,
            "current_step": self.current_step,
            "start_index": self.current_start_index,
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
        mark_to_market_reward = 0.0
        feasibility_penalty = 0.0
        excess_inventory = 0.0
        max_withdrawable_remaining = 0.0
        if is_final:
            deviation = abs(
                self.storage_level - self.storage_params.target_terminal_inventory
            )
            terminal_penalty = -self.lambda_terminal * deviation
            shaped_reward_raw = (
                raw_cashflow + terminal_penalty - self.storage_level * price
            )
        else:
            next_price = float(self.current_path[self.current_step + 1])
            remaining_steps = self.episode_length - self.current_step - 1
            max_withdrawable_remaining = (
                remaining_steps * self.storage_params.withdrawal_rate
            )
            excess_inventory = max(
                0.0,
                self.storage_level
                - self.storage_params.target_terminal_inventory
                - max_withdrawable_remaining,
            )
            feasibility_penalty = -self.lambda_feasibility * excess_inventory
            mark_to_market_reward = self.storage_level * (next_price - price)
            shaped_reward_raw = mark_to_market_reward + feasibility_penalty

        economic_reward_raw = raw_cashflow + terminal_penalty
        scaled_reward = shaped_reward_raw / self.reward_scale
        info = {
            "requested_action": requested_action,
            "executed_action": executed_action,
            "storage_level": self.storage_level,
            "previous_storage_level": previous_level,
            "target_terminal_inventory": self.storage_params.target_terminal_inventory,
            "price": price,
            "reward_scale": self.reward_scale,
            "raw_cashflow": raw_cashflow,
            "terminal_penalty": terminal_penalty,
            "economic_reward_raw": economic_reward_raw,
            "shaped_reward_raw": shaped_reward_raw,
            "raw_reward": shaped_reward_raw,
            "scaled_reward": scaled_reward,
            "mark_to_market_reward": mark_to_market_reward,
            "feasibility_penalty": feasibility_penalty,
            "excess_inventory": excess_inventory,
            "max_withdrawable_remaining": max_withdrawable_remaining,
            "current_step": self.current_step,
            "start_index": self.current_start_index,
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
