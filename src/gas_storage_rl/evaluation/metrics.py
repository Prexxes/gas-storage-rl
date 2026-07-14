"""Evaluation metrics for storage valuation experiments."""

from __future__ import annotations

import numpy as np


def interquartile_mean(values: np.ndarray) -> float:
    """Returns the mean of observations inside the interquartile range.
    
    Args:
        values: Numeric values to transform or summarize.
    
    Returns:
        Computed result.

    """
    values = np.asarray(values, dtype=np.float64)
    lower, upper = np.quantile(values, [0.25, 0.75])
    middle = values[(values >= lower) & (values <= upper)]
    return float(np.mean(middle)) if len(middle) else float(np.mean(values))


def aulc(steps: np.ndarray, returns: np.ndarray) -> float:
    """Computes area under the validation learning curve.
    
    Args:
        steps: Steps value.
        returns: Returns value.
    
    Returns:
        Area under the learning curve.

    """
    steps = np.asarray(steps, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    if len(steps) < 2:
        return 0.0
    order = np.argsort(steps)
    return float(np.trapezoid(returns[order], steps[order]))


def validation_return_aulc(
    evaluation_rows: list[dict],
    total_timesteps: int,
) -> dict[str, float]:
    """Computes validation-return AULC metrics from evaluation rows.
    
    If multiple rows share the same training step, the last row is used. This
    makes the final post-training validation replace the callback validation at
    the same step.
    
    Args:
        evaluation_rows: Evaluation rows value.
        total_timesteps: Total timesteps value.
    
    Returns:
        Validation-return AULC metrics.

    """
    step_to_return = {}
    for row in evaluation_rows:
        step = int(row["total_training_env_steps"])
        step_to_return[step] = float(row["mean_return_raw"])
    if not step_to_return:
        raw_aulc = 0.0
    else:
        steps = np.array(list(step_to_return.keys()), dtype=np.float64)
        returns = np.array(list(step_to_return.values()), dtype=np.float64)
        raw_aulc = aulc(steps, returns)
    normalized_aulc = (
        raw_aulc / float(total_timesteps) if total_timesteps > 0 else 0.0
    )
    return {
        "AULC_validation_return_raw": raw_aulc,
        "normalized_AULC_validation_return_raw": normalized_aulc,
    }


def add_risk_adjusted_return(
    metrics: dict,
    std_penalty: float,
) -> dict:
    """Adds a mean-minus-volatility validation score to metrics.
    
    Args:
        metrics: Metrics value.
        std_penalty: Std penalty value.
    
    Returns:
        Computed result.

    """
    mean_return = float(metrics["mean_return_raw"])
    std_return = float(metrics["std_return_raw"])
    metrics["risk_adjusted_return_raw"] = mean_return - float(std_penalty) * std_return
    metrics["risk_adjusted_std_penalty"] = float(std_penalty)
    return metrics


def summarize_episode_infos(infos: list[dict]) -> dict[str, float]:
    """Summarizes per-step info dictionaries for one episode.
    
    Args:
        infos: Per-step environment information dictionaries.
    
    Returns:
        Summary metrics computed from episode infos.

    """
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
    """Summarizes multiple episode results.
    
    Args:
        episode_summaries: Episode summaries value.
        split: Dataset split name.
        total_training_env_steps: Total training env steps value.
    
    Returns:
        Aggregate evaluation metrics.

    """
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
