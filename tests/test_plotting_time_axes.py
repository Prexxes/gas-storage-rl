"""Tests for trajectory plot time axes."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from gas_storage_rl.plotting import policy_plots, price_plots


def test_policy_time_values_use_absolute_simulation_day() -> None:
    """Absolute policy plot x-values include the episode start index."""
    infos = [
        {"current_step": 0, "start_index": 165},
        {"current_step": 1, "start_index": 165},
        {"current_step": 2, "start_index": 165},
    ]

    values = policy_plots._time_values(infos, "absolute")

    np.testing.assert_array_equal(values, np.array([165, 166, 167]))


def test_policy_time_values_use_day_of_year() -> None:
    """Seasonal policy plot x-values wrap to one-based day-of-year values."""
    infos = [
        {"current_step": 0, "start_index": 364},
        {"current_step": 1, "start_index": 364},
        {"current_step": 2, "start_index": 364},
    ]

    values = policy_plots._time_values(infos, "day_of_year")

    np.testing.assert_array_equal(values, np.array([365, 1, 2]))


def test_price_time_values_use_absolute_simulation_day() -> None:
    """Absolute price plot x-values include the episode start index."""
    values = price_plots._time_values(3, 66, "absolute")

    np.testing.assert_array_equal(values, np.array([66, 67, 68]))


def test_price_time_values_use_day_of_year() -> None:
    """Seasonal price plot x-values wrap to one-based day-of-year values."""
    values = price_plots._time_values(3, 364, "day_of_year")

    np.testing.assert_array_equal(values, np.array([365, 1, 2]))
