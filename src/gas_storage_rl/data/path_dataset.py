"""Fixed train/validation/test path datasets and price-path caching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gas_storage_rl.prices.historical_data import (
    assert_date_range,
    load_historical_price_csv,
)
from gas_storage_rl.prices.generators import (
    PriceGeneratorConfig,
    generate_dataset,
    split_seeds,
)


@dataclass
class PathDataset:
    """Container for fixed price paths and their split seeds."""

    paths_by_split: dict[str, np.ndarray]
    seeds_by_split: dict[str, int]
    date_ranges_by_split: dict[str, list[dict[str, str]]] | None = None

    def get_paths(self, split: str) -> np.ndarray:
        """Returns paths for a split."""
        return self.paths_by_split[split]

    def sample_path_id(self, split: str, rng: np.random.Generator) -> int:
        """Samples a path id from a split."""
        return int(rng.integers(0, len(self.paths_by_split[split])))

    def get_path(self, split: str, path_id: int) -> np.ndarray:
        """Returns a single price path."""
        return self.paths_by_split[split][path_id]

    @property
    def episode_length(self) -> int:
        """Returns the number of decision steps per path."""
        first_split = next(iter(self.paths_by_split.values()))
        return int(first_split.shape[1])


def compute_dataset_hash(config: PriceGeneratorConfig) -> str:
    """Computes a stable hash for generated price paths.

    Args:
        config: Price generation configuration.

    Returns:
        Short deterministic dataset hash.
    """
    payload = {
        "environment_name": config.environment_name,
        "episode_length": config.episode_length,
        "n_pretrain_paths": config.n_pretrain_paths,
        "n_train_paths": config.n_train_paths,
        "n_validation_paths": config.n_validation_paths,
        "n_test_paths": config.n_test_paths,
        "dataset_seed": config.dataset_seed,
        "price_process_config": config.params,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def compute_backtest_dataset_hash(
    daily_backtest_csv: str | Path,
    episode_length: int,
    window_stride: int,
    backtest_start_date: str,
) -> str:
    """Computes a stable hash for historical backtest windows."""
    payload = {
        "daily_backtest_csv": str(daily_backtest_csv),
        "episode_length": episode_length,
        "window_stride": window_stride,
        "backtest_start_date": backtest_start_date,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def load_or_generate_price_dataset(
    config: PriceGeneratorConfig,
    cache_dir: str | Path = "data/cache",
    use_cache: bool = True,
    force_regenerate: bool = False,
) -> PathDataset:
    """Loads cached price paths or generates and caches them.

    Args:
        config: Price generation configuration.
        cache_dir: Parent cache directory.
        use_cache: Whether to read/write persistent cache files.
        force_regenerate: Whether to overwrite an existing matching cache.

    Returns:
        Path dataset with train, validation, and test splits.
    """
    seeds_by_split = split_seeds(
        config.dataset_seed,
        include_pretrain=config.n_pretrain_paths > 0,
    )
    dataset_hash = compute_dataset_hash(config)
    dataset_dir = Path(cache_dir) / dataset_hash
    expected_files = [dataset_dir / f"{split}.npy" for split in seeds_by_split]

    if (
        use_cache
        and not force_regenerate
        and all(path.exists() for path in expected_files)
    ):
        paths_by_split = {
            split: np.load(dataset_dir / f"{split}.npy")
            for split in seeds_by_split
        }
        return PathDataset(paths_by_split, seeds_by_split)

    paths_by_split = generate_dataset(config)
    if use_cache:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for split, paths in paths_by_split.items():
            np.save(dataset_dir / f"{split}.npy", paths)
        _write_metadata(
            dataset_dir,
            config,
            dataset_hash,
            seeds_by_split,
            paths_by_split,
        )
    return PathDataset(paths_by_split, seeds_by_split)


def _write_metadata(
    dataset_dir: Path,
    config: PriceGeneratorConfig,
    dataset_hash: str,
    seeds_by_split: dict[str, int],
    paths_by_split: dict[str, np.ndarray],
) -> None:
    """Writes cache metadata."""
    metadata: dict[str, Any] = {
        "dataset_hash": dataset_hash,
        "environment_name": config.environment_name,
        "episode_length": config.episode_length,
        "n_pretrain_paths": config.n_pretrain_paths,
        "n_train_paths": config.n_train_paths,
        "n_validation_paths": config.n_validation_paths,
        "n_test_paths": config.n_test_paths,
        "dataset_seed": config.dataset_seed,
        "seeds_by_split": seeds_by_split,
        "price_process_config": config.params,
        "shapes": {
            split: list(paths.shape)
            for split, paths in paths_by_split.items()
        },
    }
    with (dataset_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def build_historical_backtest_paths(
    daily_backtest_csv: str | Path,
    *,
    episode_length: int,
    window_stride: int = 1,
    backtest_start_date: str = "2025-01-01",
    daily_price_column: str | None = None,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Builds rolling historical backtest windows from held-out daily prices.

    Args:
        daily_backtest_csv: Daily held-out backtest CSV.
        episode_length: Number of observations per backtest episode.
        window_stride: Step size between rolling windows.
        backtest_start_date: Inclusive first date allowed in backtesting.
        daily_price_column: Optional daily price column override.

    Returns:
        Pair of price path matrix and per-window date ranges.
    """
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if window_stride <= 0:
        raise ValueError("window_stride must be positive")
    series = load_historical_price_csv(
        daily_backtest_csv,
        price_column=daily_price_column,
        expected_split="historical_backtest",
    )
    assert_date_range(series, min_date=backtest_start_date)
    prices = series.prices.to_numpy(dtype=np.float32)
    dates = series.dates
    if len(prices) < episode_length:
        raise ValueError("Backtest series is shorter than episode_length")

    windows = []
    date_ranges = []
    for start in range(0, len(prices) - episode_length + 1, window_stride):
        end = start + episode_length
        windows.append(prices[start:end])
        date_ranges.append(
            {
                "start_date": str(dates.iloc[start].date()),
                "end_date": str(dates.iloc[end - 1].date()),
            }
        )
    return np.asarray(windows, dtype=np.float32), date_ranges


def load_or_generate_historical_backtest_dataset(
    daily_backtest_csv: str | Path,
    *,
    episode_length: int,
    cache_dir: str | Path = "data/cache/backtest",
    use_cache: bool = True,
    force_regenerate: bool = False,
    window_stride: int = 1,
    backtest_start_date: str = "2025-01-01",
    daily_price_column: str | None = None,
) -> PathDataset:
    """Loads or caches historical backtest rolling-window paths."""
    dataset_hash = compute_backtest_dataset_hash(
        daily_backtest_csv=daily_backtest_csv,
        episode_length=episode_length,
        window_stride=window_stride,
        backtest_start_date=backtest_start_date,
    )
    dataset_dir = Path(cache_dir) / dataset_hash
    paths_file = dataset_dir / "backtest.npy"
    metadata_file = dataset_dir / "metadata.json"

    if (
        use_cache
        and not force_regenerate
        and paths_file.exists()
        and metadata_file.exists()
    ):
        with metadata_file.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        return PathDataset(
            {"backtest": np.load(paths_file)},
            {"backtest": 0},
            {"backtest": metadata["date_ranges"]},
        )

    paths, date_ranges = build_historical_backtest_paths(
        daily_backtest_csv,
        episode_length=episode_length,
        window_stride=window_stride,
        backtest_start_date=backtest_start_date,
        daily_price_column=daily_price_column,
    )
    if use_cache:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.save(paths_file, paths)
        metadata = {
            "dataset_hash": dataset_hash,
            "daily_backtest_csv": str(daily_backtest_csv),
            "episode_length": episode_length,
            "window_stride": window_stride,
            "backtest_start_date": backtest_start_date,
            "split": "historical_backtest",
            "shape": list(paths.shape),
            "date_ranges": date_ranges,
        }
        with metadata_file.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
    return PathDataset({"backtest": paths}, {"backtest": 0}, {"backtest": date_ranges})
