"""Tests for policy comparison diagnostics."""

import matplotlib.pyplot as plt
import numpy as np

from gas_storage_rl.plotting.plot_policy_comparison import plot_policy_comparison


def test_policy_comparison_uses_action_heatmap() -> None:
    """Bang-bang actions are rendered as a method-by-step heatmap."""
    ppo_infos = _infos([-1.0, 1.0, -1.0])
    ppo_infos[0]["start_index"] = 125
    trajectories = [
        {"method": "ppo", "infos": ppo_infos},
        {"method": "lsmc", "infos": _infos([1.0, 0.0, -1.0])},
    ]

    figure = plot_policy_comparison(trajectories)
    action_axis = figure.axes[1]
    heatmap = action_axis.collections[0]

    assert heatmap.get_clim() == (-1.0, 1.0)
    assert [label.get_text() for label in action_axis.get_yticklabels()] == [
        "ppo",
        "lsmc",
    ]
    assert action_axis.get_ylabel() == "Method"
    assert figure.axes[0].get_legend().get_texts()[0].get_text() == (
        "Spot price (start index: 125)"
    )
    plt.close(figure)


def test_action_colorbar_does_not_shrink_heatmap_width() -> None:
    """All timestep axes have equal width and the colorbar starts to their right."""
    trajectories = [
        {"method": "ppo", "infos": _infos([-1.0, 1.0, -1.0])},
        {"method": "perfect_foresight", "infos": _infos([1.0, 0.0, -1.0])},
    ]

    figure = plot_policy_comparison(trajectories)
    figure.canvas.draw()
    data_axes = figure.axes[:4]
    colorbar_axis = figure.axes[4]
    widths = [axis.get_position().width for axis in data_axes]

    np.testing.assert_allclose(widths, widths[0])
    assert colorbar_axis.get_position().x0 > data_axes[1].get_position().x1
    plt.close(figure)


def test_action_heatmap_shows_requested_action_above_executed_action() -> None:
    """Each method row shows requested action above executed action."""
    infos = _infos([0.5, -1.0])
    infos[0]["requested_action"] = 1.0
    trajectories = [{"method": "ppo", "infos": infos}]

    figure = plot_policy_comparison(trajectories)
    action_axis = figure.axes[1]
    heatmap = action_axis.collections[0]

    np.testing.assert_allclose(heatmap.get_array(), [[1.0, -1.0], [0.5, -1.0]])
    assert len(action_axis.collections) == 2
    separator = action_axis.collections[1]
    np.testing.assert_allclose(
        separator.get_colors()[0][:3], [1.0, 0.8431372549019608, 0.0]
    )
    assert separator.get_linewidths()[0] == 3.0
    np.testing.assert_allclose(
        separator.get_segments(), [[[-0.5, 0.5], [0.5, 0.5]]]
    )
    assert action_axis.get_legend().get_texts()[0].get_text() == "Action clipped"
    plt.close(figure)


def test_policy_comparison_plots_cumulative_raw_return() -> None:
    """The final diagnostic value includes the terminal penalty."""
    infos = _infos([1.0, -1.0])
    infos[-1]["terminal_penalty"] = -5.0
    trajectories = [{"method": "ppo", "infos": infos}]

    figure = plot_policy_comparison(trajectories)
    return_axis = figure.axes[3]
    plotted_return = return_axis.lines[0].get_ydata()

    np.testing.assert_allclose(plotted_return, [-10.0, -4.0])
    assert return_axis.get_ylabel() == "Cumulative raw return"
    plt.close(figure)


def _infos(actions: list[float]) -> list[dict[str, float | int]]:
    """Returns a minimal trajectory for plotting."""
    return [
        {
            "current_step": step,
            "price": 10.0 + step,
            "executed_action": action,
            "storage_level": 1.0 + action,
            "raw_cashflow": -action * (10.0 + step),
            "terminal_penalty": 0.0,
        }
        for step, action in enumerate(actions)
    ]
