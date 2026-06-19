"""Data splitting strategies for honest evaluation."""

import warnings
from typing import Iterator, Union

import numpy as np
import numpy.typing as npt


def temporal_holdout(
    data: npt.NDArray[np.floating],
    test_frac: float = 0.2,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Temporal train/test split (last fraction for test).

    This is the DEFAULT split for time-series forecasting evaluation.
    The test set contains the most recent observations (chronologically last).

    Args:
        data: Time series data (n_samples, ...)
        test_frac: Fraction of data for test set (default: 0.2)

    Returns:
        (train, test) arrays

    Example:
        >>> data = np.arange(100).reshape(-1, 1)
        >>> train, test = temporal_holdout(data, test_frac=0.2)
        >>> len(train), len(test)
        (80, 20)
        >>> train[-1, 0] < test[0, 0]  # Train comes before test
        True
    """
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    n = len(data)
    split_idx = int(n * (1 - test_frac))

    return data[:split_idx], data[split_idx:]


def rolling_origin(
    data: npt.NDArray[np.floating],
    n_folds: int = 5,
    min_train: int = 50,
    horizon: int = 10,
    expanding: bool = True,
) -> Iterator[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]:
    """
    Rolling-origin cross-validation for time series (expanding or sliding window).

    This is the RECOMMENDED protocol for reporting forecasting skill.
    Yields (train, test) pairs where test is always a fixed horizon ahead.

    Args:
        data: Time series data (n_samples, ...)
        n_folds: Number of folds/windows
        min_train: Minimum training size
        horizon: Forecast horizon (test size)
        expanding: If True, training window expands; if False, slides (fixed size)

    Yields:
        (train, test) arrays for each fold

    Example:
        >>> data = np.arange(100).reshape(-1, 1)
        >>> folds = list(rolling_origin(data, n_folds=3, min_train=50, horizon=10))
        >>> len(folds)
        3
        >>> train, test = folds[0]
        >>> len(test)
        10
    """
    if min_train + horizon > len(data):
        raise ValueError(
            f"min_train ({min_train}) + horizon ({horizon}) exceeds data length ({len(data)})"
        )

    n = len(data)
    available = n - min_train - horizon
    step = available // n_folds

    if step < horizon:
        warnings.warn(
            f"Step size ({step}) < horizon ({horizon}). Consider fewer folds.",
            UserWarning,
        )

    for i in range(n_folds):
        test_end = min_train + horizon + i * step
        test_start = test_end - horizon

        if expanding:
            train = data[:test_start]
        else:
            # Sliding window: fixed training size = min_train
            train_start = test_start - min_train
            train = data[train_start:test_start]

        test = data[test_start:test_end]

        yield train, test


def random_split(
    data: npt.NDArray[np.floating],
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Random train/test split (OPT-IN with loud warning).

    ⚠️  WARNING: This measures INTERPOLATION, not FORECASTING capability.
    For time-series forecasting, use temporal_holdout() or rolling_origin().

    Args:
        data: Data (n_samples, ...)
        test_frac: Fraction for test
        seed: Random seed

    Returns:
        (train, test) arrays (randomly shuffled)

    Example:
        >>> data = np.arange(100).reshape(-1, 1)
        >>> train, test = random_split(data, test_frac=0.2, seed=42)
        >>> len(train), len(test)
        (80, 20)
    """
    warnings.warn(
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️  RANDOM SPLIT USED FOR TIME SERIES ⚠️\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Random splits measure INTERPOLATION, not FORECASTING.\n"
        "For honest forecasting evaluation, use:\n"
        "  - temporal_holdout() (default)\n"
        "  - rolling_origin() (recommended for skill reporting)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
        UserWarning,
        stacklevel=2,
    )

    rng = np.random.RandomState(seed)
    n = len(data)
    indices = rng.permutation(n)
    split_idx = int(n * (1 - test_frac))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return data[train_idx], data[test_idx]
