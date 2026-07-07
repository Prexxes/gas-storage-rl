"""Stable-Baselines3 callbacks for logging and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.evaluate import evaluate_policy_on_paths
from gas_storage_rl.evaluation.metrics import (
    add_risk_adjusted_return,
    summarize_episode_infos,
)
from gas_storage_rl.logging.experiment_logger import ExperimentLogger
from gas_storage_rl.logging.progress import CliProgress


class TrainingLoggingCallback(BaseCallback):
    """Logs training episodes, periodic validation, and best models."""

    def __init__(
        self,
        experiment_logger: ExperimentLogger,
        eval_env: GasStorageEnv,
        eval_freq: int,
        algorithm_name: str,
        deterministic: bool = True,
        risk_adjusted_std_penalty: float = 0.5,
        validation_path_ids: list[int] | None = None,
        total_timesteps: int | None = None,
        progress: CliProgress | None = None,
    ) -> None:
        """Initializes the callback."""
        super().__init__()
        self.experiment_logger = experiment_logger
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.algorithm_name = algorithm_name
        self.deterministic = deterministic
        self.risk_adjusted_std_penalty = float(risk_adjusted_std_penalty)
        self.validation_path_ids = validation_path_ids
        self.total_timesteps = total_timesteps
        self.progress = progress
        self.best_validation_return = float("-inf")
        self.best_risk_adjusted_validation_return = float("-inf")
        self.last_validation_step: int | None = None
        self.episode_infos: list[list[dict[str, Any]]] = [[]]
        self.episode_count = 0

    def _on_training_start(self) -> None:
        """Runs an initial validation before policy updates."""
        if self.progress is not None:
            self.progress.update(0, "initial validation")
        self._run_validation_if_due(force=True)

    def _on_step(self) -> bool:
        """Collects per-step infos and runs periodic validation."""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        self._ensure_episode_buffers(len(infos))
        for env_index, info in enumerate(infos):
            if info:
                self.episode_infos[env_index].append(info)
            if env_index < len(dones) and dones[env_index]:
                self._log_completed_episode(env_index)
        self._run_validation_if_due()
        if self.progress is not None and self.total_timesteps:
            interval = max(1, self.total_timesteps // 100)
            if (
                self.num_timesteps == self.total_timesteps
                or self.num_timesteps % interval == 0
            ):
                self.progress.update(self.num_timesteps, "training")
        return True

    def _on_training_end(self) -> None:
        """Marks CLI progress complete when training ends."""
        if self.progress is not None:
            self.progress.finish("training complete")

    def _ensure_episode_buffers(self, n_envs: int) -> None:
        """Ensures one info buffer exists per vectorized environment."""
        while len(self.episode_infos) < n_envs:
            self.episode_infos.append([])

    def _log_completed_episode(self, env_index: int) -> None:
        """Writes one completed training episode to metrics.csv."""
        infos = self.episode_infos[env_index]
        if not infos:
            return
        self.episode_count += 1
        summary = summarize_episode_infos(infos)
        summary.update(
            {
                "algorithm_name": self.algorithm_name,
                "total_env_steps": self.num_timesteps,
                "episode": self.episode_count,
            }
        )
        summary.update(self._sb3_diagnostics())
        self.experiment_logger.append_csv("metrics.csv", summary)
        self.episode_infos[env_index] = []

    def _run_validation_if_due(self, force: bool = False) -> None:
        """Runs validation at configured training-step intervals."""
        if self.eval_freq <= 0:
            return
        if not force and self.num_timesteps % self.eval_freq != 0:
            return
        metrics, _ = evaluate_policy_on_paths(
            self.eval_env,
            self.model,
            path_ids=self.validation_path_ids,
            deterministic=self.deterministic,
            total_training_env_steps=self.num_timesteps,
        )
        metrics["algorithm_name"] = self.algorithm_name
        metrics["evaluation_phase"] = "callback"
        mean_return = float(metrics["mean_return_raw"])
        add_risk_adjusted_return(metrics, self.risk_adjusted_std_penalty)
        self.experiment_logger.append_csv("evaluations.csv", metrics)
        self.last_validation_step = self.num_timesteps
        if mean_return > self.best_validation_return:
            self.best_validation_return = mean_return
            best_model_path = (
                Path(self.experiment_logger.run_dir) / "best_validation_model"
            )
            self.model.save(best_model_path)
        risk_adjusted_return = float(metrics["risk_adjusted_return_raw"])
        if risk_adjusted_return > self.best_risk_adjusted_validation_return:
            self.best_risk_adjusted_validation_return = risk_adjusted_return
            risk_adjusted_model_path = (
                Path(self.experiment_logger.run_dir)
                / "best_risk_adjusted_validation_model"
            )
            self.model.save(risk_adjusted_model_path)

    def _sb3_diagnostics(self) -> dict[str, float]:
        """Extracts currently available SB3 logger diagnostics."""
        diagnostics = {}
        name_to_value = getattr(self.model.logger, "name_to_value", {})
        keys = [
            "train/approx_kl",
            "train/clip_fraction",
            "train/entropy_loss",
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/actor_loss",
            "train/critic_loss",
            "train/ent_coef",
            "train/learning_rate",
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
        ]
        for key in keys:
            if key in name_to_value:
                diagnostics[key.replace("/", "_")] = float(name_to_value[key])
        return diagnostics


ValidationCallback = TrainingLoggingCallback
