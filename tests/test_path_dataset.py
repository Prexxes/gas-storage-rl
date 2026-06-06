"""Tests for path dataset container."""

import numpy as np

from gas_storage_rl.data.path_dataset import (
    PathDataset,
    compute_dataset_hash,
    load_or_generate_price_dataset,
)
from gas_storage_rl.prices.generators import PriceGeneratorConfig


def test_path_dataset_accessors() -> None:
    """Dataset returns fixed paths and sampled ids."""
    paths = {"train": np.ones((2, 3)), "validation": np.ones((1, 3)), "test": np.ones((1, 3))}
    dataset = PathDataset(paths, {"train": 1, "validation": 2, "test": 3})
    assert dataset.episode_length == 3
    assert dataset.get_path("train", 0).shape == (3,)
    assert dataset.sample_path_id("train", np.random.default_rng(0)) in {0, 1}


def test_price_dataset_cache_roundtrip(tmp_path) -> None:
    """Price paths are cached and loaded from stable hash directories."""
    config = PriceGeneratorConfig(
        environment_name="ou",
        episode_length=4,
        n_pretrain_paths=1,
        n_train_paths=2,
        n_validation_paths=1,
        n_test_paths=1,
        dataset_seed=123,
    )
    dataset = load_or_generate_price_dataset(config, tmp_path)
    dataset_hash = compute_dataset_hash(config)
    assert (tmp_path / dataset_hash / "metadata.json").exists()
    assert (tmp_path / dataset_hash / "pretrain.npy").exists()
    assert (tmp_path / dataset_hash / "train.npy").exists()

    loaded = load_or_generate_price_dataset(config, tmp_path)
    assert dataset.paths_by_split["train"].shape == (2, 7)
    assert dataset.get_paths("train").shape == (2, 4)
    assert np.allclose(dataset.get_paths("pretrain"), loaded.get_paths("pretrain"))
    assert np.allclose(dataset.get_paths("train"), loaded.get_paths("train"))
    assert np.array_equal(
        dataset.get_start_indices("train"),
        loaded.get_start_indices("train"),
    )
    assert np.allclose(
        dataset.get_initial_inventories("train"),
        loaded.get_initial_inventories("train"),
    )
    assert dataset.seeds_by_split == loaded.seeds_by_split


def test_initial_inventory_distribution_is_stratified_for_training() -> None:
    """Training inventories cover the configured normal distribution."""
    config = PriceGeneratorConfig(
        environment_name="deterministic",
        episode_length=3,
        n_train_paths=1000,
        n_validation_paths=10,
        n_test_paths=10,
        storage_capacity=30.0,
        initial_inventory_mean_fraction=0.30,
        initial_inventory_std_fraction=0.05,
    )

    dataset = load_or_generate_price_dataset(config, use_cache=False)
    inventories = dataset.get_initial_inventories("train")

    assert np.isclose(np.mean(inventories), 9.0, atol=0.02)
    assert np.isclose(np.std(inventories), 1.5, atol=0.03)
    assert np.all((inventories >= 0.0) & (inventories <= 30.0))
