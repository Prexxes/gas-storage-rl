"""Price path plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_price_paths(paths: np.ndarray, n_paths: int = 50, include_mean: bool = True):
    """Plots individual and mean gas spot price paths."""
    figure, axis = plt.subplots()
    selected = paths[:n_paths]
    axis.plot(selected.T, color="tab:blue", alpha=0.15, linewidth=0.8)
    if include_mean:
        axis.plot(np.mean(selected, axis=0), color="black", linewidth=2.0)
    axis.set_xlabel("Timesteps")
    axis.set_ylabel("Gas spot price")
    return figure
