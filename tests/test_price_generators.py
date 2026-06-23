"""Tests for synthetic price generation."""

import numpy as np

from gas_storage_rl.prices.generators import (
    PriceGeneratorConfig,
    generate_dataset,
    generate_price_paths,
    split_seeds,
)
from gas_storage_rl.prices.ou_process import simulate_additive_ou_process


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


def test_deterministic_prices_follow_additive_sinusoid() -> None:
    """Synthetic deterministic prices use the configured additive sinusoid."""
    paths = generate_price_paths("deterministic", 1, 366, seed=3)
    assert paths[0, 0] == 2.0
    assert np.isclose(paths[0, 91], 3.0, atol=1e-4)
    assert np.isclose(paths[0, 274], 1.0, atol=1e-4)


def test_synthetic_ou_prices_can_be_negative() -> None:
    """Additive synthetic OU paths allow negative prices."""
    paths = generate_price_paths(
        "ou",
        1_000,
        365,
        seed=3,
        params={"ou_volatility": 1.2},
    )
    assert np.any(paths < 0.0)


def test_additive_ou_uses_annual_time_scale_by_default() -> None:
    """The default daily OU innovation matches a one-year, 365-step horizon."""
    paths = simulate_additive_ou_process(
        n_paths=100_000,
        episode_length=2,
        volatility=1.2,
        seed=4,
    )
    expected_std = 1.2 * np.sqrt(
        (1.0 - np.exp(-2.0 / 365.0)) / 2.0
    )
    assert np.isclose(np.std(paths[:, 1]), expected_std, rtol=0.02)


def test_disjoint_split_seeds() -> None:
    """Dataset split seeds are disjoint."""
    seeds = split_seeds(123, include_pretrain=True)
    assert len(set(seeds.values())) == 4


def test_dataset_shapes() -> None:
    """Generated datasets respect configured split sizes."""
    dataset = generate_dataset(
        PriceGeneratorConfig(
            environment_name="deterministic",
            episode_length=5,
            n_pretrain_paths=1,
            n_train_paths=2,
            n_validation_paths=3,
            n_test_paths=4,
        )
    )
    assert dataset["pretrain"].shape == (1, 9)
    assert dataset["train"].shape == (2, 9)
    assert dataset["validation"].shape == (3, 9)
    assert dataset["test"].shape == (4, 9)
