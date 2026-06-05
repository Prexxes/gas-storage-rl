"""Feasible rule-based storage policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RuleBasedPolicy:
    """Simple quantile-threshold storage policy."""

    low_threshold: float
    high_threshold: float
    capacity: float
    episode_length: int
    withdrawal_rate: float = 1.0
    target_inventory: float = 0.0

    @classmethod
    def from_training_prices(
        cls,
        training_prices: np.ndarray,
        capacity: float,
        episode_length: int,
        withdrawal_rate: float = 1.0,
        target_inventory: float = 0.0,
    ) -> "RuleBasedPolicy":
        """Creates thresholds from training price quantiles."""
        return cls(
            low_threshold=float(np.quantile(training_prices, 0.3)),
            high_threshold=float(np.quantile(training_prices, 0.7)),
            capacity=capacity,
            episode_length=episode_length,
            withdrawal_rate=withdrawal_rate,
            target_inventory=target_inventory,
        )

    def act(self, storage_level: float, price: float, current_step: int) -> float:
        """Returns a feasible rule action before environment clipping."""
        remaining_steps = max(self.episode_length - current_step - 1, 0)
        excess_inventory = storage_level - self.target_inventory
        if excess_inventory > remaining_steps * self.withdrawal_rate:
            return -1.0
        if price > self.high_threshold and storage_level > 0.0:
            return -1.0
        safely_liquidatable = storage_level + 1.0 <= (
            remaining_steps * self.withdrawal_rate + self.target_inventory
        )
        if price < self.low_threshold and safely_liquidatable:
            return 1.0
        return 0.0

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Returns an action compatible with Stable-Baselines3."""
        del deterministic
        storage_level = float(observation[0] * self.capacity)
        price = float(observation[1] * 50.0)
        remaining_time = float(observation[4])
        current_step = int(
            round((1.0 - remaining_time) * max(self.episode_length - 1, 0))
        )
        return np.array([self.act(storage_level, price, current_step)], dtype=np.float32), None
