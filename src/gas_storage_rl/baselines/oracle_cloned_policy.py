"""Neural baseline cloned from perfect-foresight oracle actions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _ActorNetwork(nn.Module):
    """Small deterministic actor network with bounded continuous actions."""

    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        """Initializes the actor network."""
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = int(observation_dim)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, int(hidden_size)))
            layers.append(nn.ReLU())
            input_dim = int(hidden_size)
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Returns bounded actions for a batch of observations."""
        return self.network(observations)


class OracleClonedPolicy:
    """Policy trained by supervised imitation of perfect-foresight actions."""

    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        """Initializes the policy."""
        torch.manual_seed(seed)
        self.observation_dim = int(observation_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.device = torch.device(device)
        self.model = _ActorNetwork(self.observation_dim, self.hidden_sizes).to(
            self.device
        )

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int = 0,
    ) -> list[dict[str, float | int]]:
        """Fits the policy to observation/action samples."""
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(
                "observations must have shape "
                f"(n_samples, {self.observation_dim})"
            )
        if actions.shape != (len(observations), 1):
            raise ValueError("actions must have shape (n_samples, 1)")
        if len(observations) == 0:
            raise ValueError("At least one training sample is required")

        torch.manual_seed(seed)
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32)
        dataset = TensorDataset(obs_tensor, action_tensor)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate))
        history = []
        self.model.train()
        for epoch in range(1, int(epochs) + 1):
            losses = []
            for batch_obs, batch_actions in loader:
                batch_obs = batch_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                predicted_actions = self.model(batch_obs)
                loss = torch.nn.functional.mse_loss(
                    predicted_actions,
                    batch_actions,
                )
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
        return history

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Returns an action compatible with Stable-Baselines3 policies."""
        del deterministic
        observation_array = np.asarray(observation, dtype=np.float32).reshape(
            1,
            self.observation_dim,
        )
        obs_tensor = torch.as_tensor(
            observation_array,
            dtype=torch.float32,
            device=self.device,
        )
        self.model.eval()
        with torch.no_grad():
            action = self.model(obs_tensor).cpu().numpy()[0]
        return np.asarray(action, dtype=np.float32), None

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Saves policy weights and construction metadata."""
        output_path = Path(path)
        payload = {
            "observation_dim": self.observation_dim,
            "hidden_sizes": self.hidden_sizes,
            "state_dict": self.model.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, output_path)
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "OracleClonedPolicy":
        """Loads a saved oracle-cloned policy."""
        payload = torch.load(Path(path), map_location=device)
        policy = cls(
            observation_dim=int(payload["observation_dim"]),
            hidden_sizes=tuple(payload["hidden_sizes"]),
            device=device,
        )
        policy.model.load_state_dict(payload["state_dict"])
        policy.model.eval()
        return policy
