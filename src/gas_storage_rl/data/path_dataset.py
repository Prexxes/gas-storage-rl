"""Fixed train/validation/test path datasets and price-path caching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

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
    episode_length_override: int | None = None
    start_indices_by_split: dict[str, np.ndarray] | None = None
    base_dates_by_split: dict[str, str] | None = None
    initial_inventories_by_split: dict[str, np.ndarray] | None = None

    def get_paths(self, split: str) -> np.ndarray:
        """Returns fixed episode windows for a split.
        
        Args:
            split: Dataset split name.
        
        Returns:
            Fixed episode price paths for the split.

        """
        paths = self.paths_by_split[split]
        if paths.shape[1] == self.episode_length:
            return paths
        starts = self.get_start_indices(split)
        return np.stack(
            [
                path[start : start + self.episode_length]
                for path, start in zip(paths, starts, strict=True)
            ]
        )

    def sample_path_id(self, split: str, rng: np.random.Generator) -> int:
        """Samples a path id from a split.
        
        Args:
            split: Dataset split name.
            rng: Random number generator used for deterministic sampling.
        
        Returns:
            Sampled path identifier.

        """
        return int(rng.integers(0, len(self.paths_by_split[split])))

    def get_path(self, split: str, path_id: int) -> np.ndarray:
        """Returns one fixed episode window.
        
        Args:
            split: Dataset split name.
            path_id: Identifier of the path to retrieve or evaluate.
        
        Returns:
            Selected fixed episode price path.

        """
        start = int(self.get_start_indices(split)[path_id])
        return self.get_path_window(split, path_id, start)

    def get_raw_path(self, split: str, path_id: int) -> np.ndarray:
        """Returns one stored contiguous raw price path.
        
        Args:
            split: Dataset split name.
            path_id: Identifier of the path to retrieve or evaluate.
        
        Returns:
            Stored contiguous raw price path.

        """
        return self.paths_by_split[split][path_id]

    def get_path_window(
        self,
        split: str,
        path_id: int,
        start_index: int,
    ) -> np.ndarray:
        """Returns a contiguous episode window from a stored raw path.
        
        Args:
            split: Dataset split name.
            path_id: Identifier of the path to retrieve or evaluate.
            start_index: Start index value.
        
        Returns:
            Contiguous price window for the requested episode.
        
        Raises:
            ValueError: If an input value or configuration is invalid.

        """
        raw_path = self.get_raw_path(split, path_id)
        max_start = len(raw_path) - self.episode_length
        if not 0 <= start_index <= max_start:
            raise ValueError(f"start_index must be between 0 and {max_start}")
        return raw_path[start_index : start_index + self.episode_length]

    def get_start_indices(self, split: str) -> np.ndarray:
        """Returns deterministic per-path episode start indices.
        
        Args:
            split: Dataset split name.
        
        Returns:
            Computed result.

        """
        if self.start_indices_by_split is None:
            return np.zeros(len(self.paths_by_split[split]), dtype=np.int64)
        return self.start_indices_by_split[split]

    def get_start_dates(self, split: str) -> list[date]:
        """Returns deterministic episode start dates for a split.
        
        Args:
            split: Dataset split name.
        
        Returns:
            Computed result.

        """
        if self.base_dates_by_split is not None and split in self.base_dates_by_split:
            base_date = datetime.strptime(
                self.base_dates_by_split[split], "%Y-%m-%d"
            ).date()
            return [
                base_date + timedelta(days=int(start))
                for start in self.get_start_indices(split)
            ]
        if self.date_ranges_by_split is not None:
            date_ranges = self.date_ranges_by_split.get(split)
            if date_ranges:
                return [
                    datetime.strptime(item["start_date"], "%Y-%m-%d").date()
                    for item in date_ranges
                ]
        return [date(2001, 1, 1)] * len(self.paths_by_split[split])

    def get_initial_inventories(
        self,
        split: str,
        default: float = 0.0,
    ) -> np.ndarray:
        """Returns deterministic per-path initial and target inventories.
        
        Args:
            split: Dataset split name.
            default: Default value.
        
        Returns:
            Computed result.

        """
        if self.initial_inventories_by_split is None:
            return np.full(len(self.paths_by_split[split]), default, dtype=np.float64)
        return self.initial_inventories_by_split[split]

    @property
    def episode_length(self) -> int:
        """Returns the number of decision steps per path.
        
        Returns:
            Number of decision steps in each episode.

        """
        if self.episode_length_override is not None:
            return self.episode_length_override
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
        "simulation_length": config.simulation_length,
        "rolling_window_version": 2,
        "n_pretrain_paths": config.n_pretrain_paths,
        "n_train_paths": config.n_train_paths,
        "n_validation_paths": config.n_validation_paths,
        "n_test_paths": config.n_test_paths,
        "dataset_seed": config.dataset_seed,
        "storage_capacity": config.storage_capacity,
        "initial_inventory_mean_fraction": config.initial_inventory_mean_fraction,
        "initial_inventory_std_fraction": config.initial_inventory_std_fraction,
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
    """Computes a stable hash for historical backtest windows.
    
    Args:
        daily_backtest_csv: Daily backtest csv value.
        episode_length: Episode length value.
        window_stride: Window stride value.
        backtest_start_date: Backtest start date value.
    
    Returns:
        Stable hash for the backtest dataset.

    """
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
    metadata_file = dataset_dir / "metadata.json"

    if (
        use_cache
        and not force_regenerate
        and all(path.exists() for path in expected_files)
        and metadata_file.exists()
    ):
        with metadata_file.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        paths_by_split = {
            split: np.load(dataset_dir / f"{split}.npy")
            for split in seeds_by_split
        }
        start_indices_by_split = {
            split: np.asarray(metadata["start_indices_by_split"][split], dtype=np.int64)
            for split in seeds_by_split
        }
        initial_inventories_by_split = {
            split: np.asarray(
                metadata["initial_inventories_by_split"][split],
                dtype=np.float64,
            )
            for split in seeds_by_split
        }
        return PathDataset(
            paths_by_split,
            seeds_by_split,
            date_ranges_by_split=metadata["date_ranges_by_split"],
            episode_length_override=config.episode_length,
            start_indices_by_split=start_indices_by_split,
            base_dates_by_split=metadata["base_dates_by_split"],
            initial_inventories_by_split=initial_inventories_by_split,
        )

    paths_by_split = generate_dataset(config)
    start_indices_by_split = _generate_start_indices(config, paths_by_split, seeds_by_split)
    base_dates_by_split = {split: "2001-01-01" for split in paths_by_split}
    date_ranges_by_split = _build_date_ranges(
        config.episode_length,
        start_indices_by_split,
        base_dates_by_split,
    )
    initial_inventories_by_split = _generate_initial_inventories(
        config,
        paths_by_split,
        seeds_by_split,
    )
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
            start_indices_by_split,
            base_dates_by_split,
            date_ranges_by_split,
            initial_inventories_by_split,
        )
    return PathDataset(
        paths_by_split,
        seeds_by_split,
        date_ranges_by_split=date_ranges_by_split,
        episode_length_override=config.episode_length,
        start_indices_by_split=start_indices_by_split,
        base_dates_by_split=base_dates_by_split,
        initial_inventories_by_split=initial_inventories_by_split,
    )


def _write_metadata(
    dataset_dir: Path,
    config: PriceGeneratorConfig,
    dataset_hash: str,
    seeds_by_split: dict[str, int],
    paths_by_split: dict[str, np.ndarray],
    start_indices_by_split: dict[str, np.ndarray],
    base_dates_by_split: dict[str, str],
    date_ranges_by_split: dict[str, list[dict[str, str]]],
    initial_inventories_by_split: dict[str, np.ndarray],
) -> None:
    """Writes cache metadata.
    
    Args:
        dataset_dir: Dataset dir value.
        config: Experiment configuration dictionary.
        dataset_hash: Dataset hash value.
        seeds_by_split: Seeds by split value.
        paths_by_split: Paths by split value.
        start_indices_by_split: Start indices by split value.
        base_dates_by_split: Base dates by split value.
        date_ranges_by_split: Date ranges by split value.
        initial_inventories_by_split: Initial inventories by split value.

    """
    metadata: dict[str, Any] = {
        "dataset_hash": dataset_hash,
        "environment_name": config.environment_name,
        "episode_length": config.episode_length,
        "simulation_length": config.simulation_length,
        "n_pretrain_paths": config.n_pretrain_paths,
        "n_train_paths": config.n_train_paths,
        "n_validation_paths": config.n_validation_paths,
        "n_test_paths": config.n_test_paths,
        "dataset_seed": config.dataset_seed,
        "storage_capacity": config.storage_capacity,
        "initial_inventory_mean_fraction": config.initial_inventory_mean_fraction,
        "initial_inventory_std_fraction": config.initial_inventory_std_fraction,
        "seeds_by_split": seeds_by_split,
        "price_process_config": config.params,
        "start_indices_by_split": {
            split: starts.astype(int).tolist()
            for split, starts in start_indices_by_split.items()
        },
        "base_dates_by_split": base_dates_by_split,
        "date_ranges_by_split": date_ranges_by_split,
        "initial_inventories_by_split": {
            split: inventories.astype(float).tolist()
            for split, inventories in initial_inventories_by_split.items()
        },
        "shapes": {
            split: list(paths.shape)
            for split, paths in paths_by_split.items()
        },
    }
    with (dataset_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def _generate_start_indices(
    config: PriceGeneratorConfig,
    paths_by_split: dict[str, np.ndarray],
    seeds_by_split: dict[str, int],
) -> dict[str, np.ndarray]:
    """Generates deterministic fixed starts for evaluation and baselines.
    
    Args:
        config: Experiment configuration dictionary.
        paths_by_split: Paths by split value.
        seeds_by_split: Seeds by split value.
    
    Returns:
        Generate start indices result.

    """
    max_start = config.simulation_length - config.episode_length
    return {
        split: np.random.default_rng(seeds_by_split[split] + 4_000_003).integers(
            0,
            max_start + 1,
            size=len(paths),
            dtype=np.int64,
        )
        for split, paths in paths_by_split.items()
    }


def _generate_initial_inventories(
    config: PriceGeneratorConfig,
    paths_by_split: dict[str, np.ndarray],
    seeds_by_split: dict[str, int],
) -> dict[str, np.ndarray]:
    """Generates fixed per-path initial inventories for non-training workflows.
    
    Args:
        config: Experiment configuration dictionary.
        paths_by_split: Paths by split value.
        seeds_by_split: Seeds by split value.
    
    Returns:
        Generate initial inventories result.

    """
    output = {}
    for split, paths in paths_by_split.items():
        rng = np.random.default_rng(seeds_by_split[split] + 5_000_003)
        if config.initial_inventory_std_fraction == 0.0:
            fractions = np.full(
                len(paths),
                config.initial_inventory_mean_fraction,
                dtype=np.float64,
            )
        elif split in {"pretrain", "train"}:
            probabilities = (np.arange(len(paths), dtype=np.float64) + 0.5) / len(
                paths
            )
            fractions = norm.ppf(
                probabilities,
                loc=config.initial_inventory_mean_fraction,
                scale=config.initial_inventory_std_fraction,
            )
            rng.shuffle(fractions)
        else:
            fractions = rng.normal(
                loc=config.initial_inventory_mean_fraction,
                scale=config.initial_inventory_std_fraction,
                size=len(paths),
            )
        output[split] = (
            np.clip(fractions, 0.0, 1.0) * config.storage_capacity
        ).astype(np.float64)
    return output


def _build_date_ranges(
    episode_length: int,
    start_indices_by_split: dict[str, np.ndarray],
    base_dates_by_split: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    """Builds fixed episode date ranges from raw-path start offsets.
    
    Args:
        episode_length: Episode length value.
        start_indices_by_split: Start indices by split value.
        base_dates_by_split: Base dates by split value.
    
    Returns:
        Build date ranges result.

    """
    output = {}
    for split, starts in start_indices_by_split.items():
        base_date = datetime.strptime(base_dates_by_split[split], "%Y-%m-%d").date()
        output[split] = [
            {
                "start_date": str(base_date + timedelta(days=int(start))),
                "end_date": str(
                    base_date + timedelta(days=int(start) + episode_length - 1)
                ),
            }
            for start in starts
        ]
    return output


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
    
    Raises:
        ValueError: If an input value or configuration is invalid.

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
    """Loads or caches historical backtest rolling-window paths.
    
    Args:
        daily_backtest_csv: Daily backtest csv value.
        episode_length: Episode length value.
        cache_dir: Cache dir value.
        use_cache: Use cache value.
        force_regenerate: Force regenerate value.
        window_stride: Window stride value.
        backtest_start_date: Backtest start date value.
        daily_price_column: Daily price column value.
    
    Returns:
        Computed result.

    """
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
