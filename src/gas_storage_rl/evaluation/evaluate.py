"""Policy evaluation on fixed path splits."""

from __future__ import annotations

from typing import Any

from gas_storage_rl.envs.gas_storage_env import GasStorageEnv
from gas_storage_rl.evaluation.metrics import summarize_episode_infos, summarize_evaluation


def evaluate_policy_on_paths(
    env: GasStorageEnv,
    policy: Any,
    path_ids: list[int] | None = None,
    deterministic: bool = True,
    total_training_env_steps: int = 0,
) -> tuple[dict, list[dict]]:
    """Evaluates a policy on fixed path ids."""
    n_paths = len(env.dataset.get_paths(env.split))
    selected = path_ids if path_ids is not None else list(range(n_paths))
    episode_summaries = []
    trajectories = []
    for path_id in selected:
        obs, _ = env.reset(options={"path_id": path_id})
        done = False
        infos = []
        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            infos.append(info)
            done = terminated or truncated
        episode_summaries.append(summarize_episode_infos(infos))
        trajectories.append({"path_id": path_id, "infos": infos})
    return (
        summarize_evaluation(
            episode_summaries,
            env.split,
            total_training_env_steps,
        ),
        trajectories,
    )
