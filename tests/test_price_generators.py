"""Tests for synthetic price generation."""

import numpy as np

from gas_storage_rl.prices.generators import (
    PriceGeneratorConfig,
    generate_dataset,
    generate_price_paths,
    split_seeds,
)


def test_deterministic_price_reproducibility_and_repeated_paths() -> None:
    """Deterministic paths are reproducible and repeated."""
    paths = generate_price_paths("deterministic", 3, 12, seed=1)
    paths_again = generate_price_paths("deterministic", 3, 12, seed=99)
    assert np.allclose(paths, paths_again)
    assert np.allclose(paths[0], paths[1])


def test_ou_and_jump_reproducibility_with_fixed_seed() -> None:
    """OU and jump paths are reproducible with fixed seeds."""
    assert np.allclose(
        generate_price_paths("ou", 2, 12, seed=7),
        generate_price_paths("ou", 2, 12, seed=7),
    )
    assert np.allclose(
        generate_price_paths("jump", 2, 12, seed=8),
        generate_price_paths("jump", 2, 12, seed=8),
    )


def test_positive_prices() -> None:
    """Generated prices are strictly positive."""
    paths = generate_price_paths("jump", 10, 20, seed=3)
    assert np.all(paths > 0.0)


def test_disjoint_split_seeds() -> None:
    """Dataset split seeds are disjoint."""
    seeds = split_seeds(123)
    assert len(set(seeds.values())) == 3


def test_dataset_shapes() -> None:
    """Generated datasets respect configured split sizes."""
    dataset = generate_dataset(
        PriceGeneratorConfig(
            environment_name="deterministic",
            episode_length=5,
            n_train_paths=2,
            n_validation_paths=3,
            n_test_paths=4,
        )
    )
    assert dataset["train"].shape == (2, 5)
    assert dataset["validation"].shape == (3, 5)
    assert dataset["test"].shape == (4, 5)
