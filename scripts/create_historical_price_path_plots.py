"""Create price-path plots from the historical calibration pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gas_storage_rl.data.path_dataset import build_historical_backtest_paths
from gas_storage_rl.prices.calibration import calibrate_historical_price_process
from gas_storage_rl.prices.generators import generate_price_paths
from gas_storage_rl.training.config import load_config


def main() -> None:
    """Writes plots for calibrated synthetic and historical backtest paths."""
    config = load_config("configs/historical_debug.yaml")
    env_config = config["environment_config"]
    historical_config = config["historical_data_config"]
    calibrated_config = config["calibrated_price_process_config"]
    out_dir = Path("reports/historical_price_path_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = calibrate_historical_price_process(
        historical_config["monthly_calibration_csv"],
        historical_config["daily_calibration_csv"],
        calibration_end_date=historical_config["calibration_end_date"],
        jump_threshold_sigma=float(calibrated_config["jump_threshold_sigma"]),
    )
    synthetic_paths = generate_price_paths(
        environment_name="historical_calibrated",
        n_paths=80,
        episode_length=int(env_config["episode_length"]),
        seed=int(config["seeds"]["dataset_seed"]),
        params=calibration.to_price_params(),
    )
    _plot_paths(
        synthetic_paths,
        "Historically calibrated synthetic price paths",
        out_dir / "historical_calibrated_synthetic_price_paths.png",
    )

    backtest_paths, _ = build_historical_backtest_paths(
        historical_config["daily_backtest_csv"],
        episode_length=int(env_config["episode_length"]),
        window_stride=int(calibrated_config["backtest_window_stride"]),
        backtest_start_date=historical_config["backtest_start_date"],
    )
    _plot_paths(
        backtest_paths,
        "Historical backtest rolling-window price paths",
        out_dir / "historical_backtest_price_paths.png",
    )
    print(out_dir.resolve())


def _plot_paths(paths: np.ndarray, title: str, output_path: Path) -> None:
    """Writes one price-path plot."""
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=140)
    selected = paths[: min(50, len(paths))]
    axis.plot(selected.T, color="#2f6f9f", alpha=0.16, linewidth=0.9)
    axis.plot(
        np.mean(selected, axis=0),
        color="#111111",
        linewidth=2.2,
        label="Mean",
    )
    axis.set_title(title)
    axis.set_xlabel("Day")
    axis.set_ylabel("Gas spot price")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
