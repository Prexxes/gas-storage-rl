"""Statistical helpers shared by plotting modules."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_percentile_ci(
    values: np.ndarray,
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10_000,
    random_seed: int = 0,
) -> tuple[float, float]:
    """Returns a percentile bootstrap confidence interval for the sample mean.

    Args:
        values: One-dimensional sample values.
        confidence_level: Confidence mass covered by the interval.
        n_bootstrap: Number of bootstrap resamples.
        random_seed: Seed used for deterministic resampling.

    Returns:
        Lower and upper confidence interval bounds.

    Raises:
        ValueError: If inputs are empty or outside valid ranges.
    """
    return bootstrap_percentile_statistic_ci(
        values,
        statistic=np.mean,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )


def bootstrap_percentile_statistic_ci(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 10_000,
    random_seed: int = 0,
) -> tuple[float, float]:
    """Returns a percentile bootstrap confidence interval for a statistic.

    Args:
        values: One-dimensional sample values.
        statistic: Statistic function applied to each bootstrap resample.
        confidence_level: Confidence mass covered by the interval.
        n_bootstrap: Number of bootstrap resamples.
        random_seed: Seed used for deterministic resampling.

    Returns:
        Lower and upper confidence interval bounds.

    Raises:
        ValueError: If inputs are empty or outside valid ranges.
    """
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1:
        raise ValueError("values must be one-dimensional")
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("values must contain at least one finite value")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if sample.size == 1:
        value = float(sample[0])
        return value, value

    rng = np.random.default_rng(random_seed)
    draws = rng.choice(sample, size=(int(n_bootstrap), sample.size), replace=True)
    bootstrap_statistics = np.array([float(statistic(draw)) for draw in draws])
    tail = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(bootstrap_statistics, [tail, 1.0 - tail])
    return float(lower), float(upper)
