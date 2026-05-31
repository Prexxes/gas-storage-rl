"""Build calibrated synthetic datasets and held-out historical backtest windows."""

from __future__ import annotations

import argparse
import json

from gas_storage_rl.data.path_dataset import (
    load_or_generate_historical_backtest_dataset,
)
from gas_storage_rl.training.config import build_environment, load_config


def main() -> None:
    """Builds configured historical calibration and backtest datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    synthetic_dataset, _, _ = build_environment(config)

    historical_config = config["historical_data_config"]
    dataset_config = config["dataset_config"]
    calibrated_config = config.get("calibrated_price_process_config", {})
    backtest_dataset = load_or_generate_historical_backtest_dataset(
        historical_config["daily_backtest_csv"],
        episode_length=int(config["environment_config"]["episode_length"]),
        cache_dir=dataset_config.get("backtest_cache_dir", "data/cache/backtest"),
        use_cache=bool(dataset_config.get("use_cache", True)),
        force_regenerate=bool(dataset_config.get("force_regenerate", False)),
        window_stride=int(calibrated_config.get("backtest_window_stride", 1)),
        backtest_start_date=historical_config.get("backtest_start_date", "2025-01-01"),
        daily_price_column=historical_config.get("daily_backtest_price_column"),
    )

    summary = {
        "synthetic_shapes": {
            split: list(paths.shape)
            for split, paths in synthetic_dataset.paths_by_split.items()
        },
        "backtest_shape": list(backtest_dataset.get_paths("backtest").shape),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
