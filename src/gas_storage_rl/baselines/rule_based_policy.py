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
    injection_rate: float = 1.0
    withdrawal_rate: float = 1.0
    target_inventory: float = 0.0
    price_scale: float = 50.0

    @classmethod
    def from_training_prices(
        cls,
        training_prices: np.ndarray,
        capacity: float,
        episode_length: int,
        injection_rate: float = 1.0,
        withdrawal_rate: float = 1.0,
        target_inventory: float = 0.0,
        price_scale: float = 50.0,
    ) -> "RuleBasedPolicy":
        """Creates thresholds from training price quantiles.
        
        Args:
            training_prices: Training prices value.
            capacity: Capacity value.
            episode_length: Episode length value.
            injection_rate: Injection rate value.
            withdrawal_rate: Withdrawal rate value.
            target_inventory: Target inventory value.
            price_scale: Price normalization scale used by observations.
        
        Returns:
            Computed result.

        """
        return cls(
            low_threshold=float(np.quantile(training_prices, 0.3)),
            high_threshold=float(np.quantile(training_prices, 0.7)),
            capacity=capacity,
            episode_length=episode_length,
            injection_rate=injection_rate,
            withdrawal_rate=withdrawal_rate,
            target_inventory=target_inventory,
            price_scale=price_scale,
        )

    def act(
        self,
        storage_level: float,
        price: float,
        current_step: int,
        target_inventory: float | None = None,
    ) -> float:
        """Returns a feasible rule action before environment clipping.
        
        Args:
            storage_level: Storage level value.
            price: Price value.
            current_step: Current step value.
            target_inventory: Target inventory value.
        
        Returns:
            Action selected by the policy.

        """
        target = self.target_inventory if target_inventory is None else target_inventory
        remaining_steps = max(self.episode_length - current_step - 1, 0)
        excess_inventory = storage_level - target
        if excess_inventory > remaining_steps * self.withdrawal_rate:
            return -1.0
        inventory_shortfall = target - storage_level
        if inventory_shortfall > remaining_steps * self.injection_rate:
            return 1.0
        if price > self.high_threshold and storage_level > 0.0:
            return -1.0
        safely_liquidatable = storage_level + 1.0 <= (
            remaining_steps * self.withdrawal_rate + target
        )
        if price < self.low_threshold and safely_liquidatable:
            return 1.0
        return 0.0

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Returns an action compatible with Stable-Baselines3.
        
        Args:
            observation: Observation value.
            deterministic: Deterministic value.
        
        Returns:
            Predicted action and optional recurrent state.

        """
        del deterministic
        storage_level = float(observation[0] * self.capacity)
        price = float(observation[1] * self.price_scale)
        remaining_time = float(observation[4])
        current_step = int(
            round((1.0 - remaining_time) * max(self.episode_length - 1, 0))
        )
        target_inventory = float(observation[5] * self.capacity)
        action = self.act(
            storage_level,
            price,
            current_step,
            target_inventory,
        )
        return np.array([action], dtype=np.float32), None
