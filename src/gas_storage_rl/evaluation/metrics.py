"""Evaluation metrics for storage valuation experiments."""

from __future__ import annotations

import numpy as np


def interquartile_mean(values: np.ndarray) -> float:
    """Returns the mean of observations inside the interquartile range."""
    values = np.asarray(values, dtype=np.float64)
    lower, upper = np.quantile(values, [0.25, 0.75])
    middle = values[(values >= lower) & (values <= upper)]
    return float(np.mean(middle)) if len(middle) else float(np.mean(values))


def aulc(steps: np.ndarray, returns: np.ndarray) -> float:
    """Computes area under the validation learning curve."""
    steps = np.asarray(steps, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    if len(steps) < 2:
        return 0.0
    order = np.argsort(steps)
    return float(np.trapezoid(returns[order], steps[order]))


def add_risk_adjusted_return(
    metrics: dict,
    std_penalty: float,
) -> dict:
    """Adds a mean-minus-volatility validation score to metrics."""
    mean_return = float(metrics["mean_return_raw"])
    std_return = float(metrics["std_return_raw"])
    metrics["risk_adjusted_return_raw"] = mean_return - float(std_penalty) * std_return
    metrics["risk_adjusted_std_penalty"] = float(std_penalty)
    return metrics


def summarize_episode_infos(infos: list[dict]) -> dict[str, float]:
    """Summarizes per-step info dictionaries for one episode."""
    rewards = np.array(
        [
            info.get("raw_reward", info["raw_cashflow"] + info["terminal_penalty"])
            for info in infos
        ],
        dtype=np.float64,
    )
    scaled_rewards = np.array(
        [info["scaled_reward"] for info in infos],
        dtype=np.float64,
    )
    actions = np.array([info["executed_action"] for info in infos], dtype=np.float64)
    storage = np.array([info["storage_level"] for info in infos], dtype=np.float64)
    target_inventory = float(infos[-1].get("target_terminal_inventory", 0.0))
    return {
        "episode_return_raw": float(np.sum(rewards)),
        "episode_return_scaled": float(np.sum(scaled_rewards)),
        "episode_length": float(len(infos)),
        "final_storage_level": float(storage[-1]),
        "terminal_deviation": float(abs(storage[-1] - target_inventory)),
        "terminal_penalty": float(np.sum([info["terminal_penalty"] for info in infos])),
        "cumulative_cashflow": float(np.sum([info["raw_cashflow"] for info in infos])),
        "number_of_constrained_actions": float(
            np.sum(
                [
                    abs(info["requested_action"] - info["executed_action"]) > 1e-8
                    for info in infos
                ]
            )
        ),
        "total_injected_volume": float(np.sum(np.maximum(actions, 0.0))),
        "total_withdrawn_volume": float(-np.sum(np.minimum(actions, 0.0))),
        "mean_storage_level": float(np.mean(storage)),
        "mean_action": float(np.mean(actions)),
    }


def summarize_evaluation(
    episode_summaries: list[dict],
    split: str,
    total_training_env_steps: int = 0,
) -> dict[str, float | str]:
    """Summarizes multiple episode results."""
    raw_returns = np.array([item["episode_return_raw"] for item in episode_summaries])
    return {
        "total_training_env_steps": total_training_env_steps,
        "split": split,
        "mean_return_scaled": float(
            np.mean([item["episode_return_scaled"] for item in episode_summaries])
        ),
        "mean_return_raw": float(np.mean(raw_returns)),
        "median_return_raw": float(np.median(raw_returns)),
        "std_return_raw": float(np.std(raw_returns)),
        "min_return_raw": float(np.min(raw_returns)),
        "max_return_raw": float(np.max(raw_returns)),
        "interquartile_mean_return_raw": interquartile_mean(raw_returns),
        "mean_terminal_deviation": float(
            np.mean([item["terminal_deviation"] for item in episode_summaries])
        ),
        "mean_cumulative_cashflow": float(
            np.mean([item["cumulative_cashflow"] for item in episode_summaries])
        ),
        "mean_terminal_penalty": float(
            np.mean([item["terminal_penalty"] for item in episode_summaries])
        ),
        "mean_number_of_constrained_actions": float(
            np.mean(
                [
                    item["number_of_constrained_actions"]
                    for item in episode_summaries
                ]
            )
        ),
    }
