"""Build synthetic datasets and held-out historical backtest windows."""

from __future__ import annotations

import argparse
import json

from gas_storage_rl.evaluation.backtest import build_backtest_evaluation_dataset
from gas_storage_rl.training.config import build_environment, load_config


def main() -> None:
    """Builds configured synthetic and historical backtest datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    synthetic_dataset, _, _ = build_environment(config)
    dataset = build_backtest_evaluation_dataset(config, synthetic_dataset)

    summary = {
        "synthetic_shapes": {
            split: list(paths.shape)
            for split, paths in synthetic_dataset.paths_by_split.items()
        },
        "backtest_shape": list(dataset.get_paths("backtest").shape),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
