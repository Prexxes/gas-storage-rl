"""Learning curve and return distribution plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(metrics: pd.DataFrame):
    """Plots validation return over environment steps.
    
    Args:
        metrics: Metrics value.
    
    Returns:
        Computed result.

    """
    figure, axis = plt.subplots()
    step_column = (
        "total_training_env_steps"
        if "total_training_env_steps" in metrics.columns
        else "total_env_steps"
    )
    axis.plot(metrics[step_column], metrics["mean_return_raw"])
    axis.set_xlabel("Total environment steps")
    axis.set_ylabel("Validation return")
    return figure


def plot_return_distribution(evaluation_results: pd.DataFrame):
    """Plots return distributions grouped by algorithm or benchmark.
    
    Args:
        evaluation_results: Evaluation results value.
    
    Returns:
        Computed result.

    """
    figure, axis = plt.subplots()
    groups = [
        group["episode_return_raw"].to_numpy()
        for _, group in evaluation_results.groupby("algorithm_name")
    ]
    labels = list(evaluation_results.groupby("algorithm_name").groups.keys())
    axis.boxplot(groups, labels=labels)
    axis.set_ylabel("Raw return")
    return figure
