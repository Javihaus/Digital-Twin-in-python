"""Naive baseline predictors for forecasting (mandatory for honest evaluation)."""

import numpy as np
import numpy.typing as npt


def persistence(
    train: npt.NDArray[np.floating],
    horizon: int,
) -> npt.NDArray[np.floating]:
    """
    Persistence (naive) forecast: last observed value repeated.

    Also called "random walk forecast" or "no-change forecast".
    This is the MINIMUM VIABLE baseline for any forecasting task.

    Args:
        train: Training data (n_train, n_features)
        horizon: Forecast horizon (number of steps ahead)

    Returns:
        Forecast (horizon, n_features) — last train value repeated

    Example:
        >>> train = np.array([[1], [2], [3]])
        >>> forecast = persistence(train, horizon=5)
        >>> np.all(forecast == 3)
        True
    """
    last_value = train[-1]
    return np.tile(last_value, (horizon, 1))


def drift(
    train: npt.NDArray[np.floating],
    horizon: int,
) -> npt.NDArray[np.floating]:
    """
    Drift forecast: linear extrapolation from last two points.

    Captures linear trends. Often better than persistence for trending series.

    Args:
        train: Training data (n_train, n_features), requires n_train >= 2
        horizon: Forecast horizon

    Returns:
        Forecast (horizon, n_features)

    Example:
        >>> train = np.array([[1], [2], [3]])
        >>> forecast = drift(train, horizon=2)
        >>> forecast[0, 0], forecast[1, 0]
        (4.0, 5.0)
    """
    if len(train) < 2:
        raise ValueError("Drift forecast requires at least 2 training points")

    last_value = train[-1]
    slope = train[-1] - train[-2]

    forecasts = []
    for h in range(1, horizon + 1):
        forecasts.append(last_value + h * slope)

    return np.array(forecasts)


def mean_forecast(
    train: npt.NDArray[np.floating],
    horizon: int,
) -> npt.NDArray[np.floating]:
    """
    Mean forecast: training mean repeated.

    Optimal for i.i.d. data with no trend/seasonality.

    Args:
        train: Training data (n_train, n_features)
        horizon: Forecast horizon

    Returns:
        Forecast (horizon, n_features)

    Example:
        >>> train = np.array([[1], [2], [3]])
        >>> forecast = mean_forecast(train, horizon=2)
        >>> np.all(forecast == 2.0)
        True
    """
    mean_value = train.mean(axis=0)
    return np.tile(mean_value, (horizon, 1))


def seasonal_naive(
    train: npt.NDArray[np.floating],
    horizon: int,
    period: int,
) -> npt.NDArray[np.floating]:
    """
    Seasonal naive forecast: repeat last season.

    For data with known periodicity (e.g., daily, weekly, yearly).

    Args:
        train: Training data (n_train, n_features)
        horizon: Forecast horizon
        period: Seasonal period (e.g., 7 for weekly, 365 for yearly)

    Returns:
        Forecast (horizon, n_features)

    Example:
        >>> train = np.array([[1], [2], [3], [4], [5], [6], [7]])
        >>> forecast = seasonal_naive(train, horizon=3, period=7)
        >>> # Repeats last 7 values cyclically
    """
    if len(train) < period:
        raise ValueError(f"Training data length ({len(train)}) < period ({period})")

    # Extract last full period
    last_season = train[-period:]

    # Tile cyclically
    n_repeats = int(np.ceil(horizon / period))
    tiled = np.tile(last_season, (n_repeats, 1))

    return tiled[:horizon]


def get_best_baseline(
    train: npt.NDArray[np.floating],
    test: npt.NDArray[np.floating],
    period: int | None = None,
) -> tuple[str, npt.NDArray[np.floating], float]:
    """
    Evaluate all baselines and return the best one (by RMSE).

    This is used by EvalReport to select the reference baseline for skill scores.

    Args:
        train: Training data
        test: Test data (for computing forecast target shape)
        period: Optional seasonal period

    Returns:
        (baseline_name, forecast, rmse)

    Example:
        >>> train = np.arange(100).reshape(-1, 1)
        >>> test = np.arange(100, 110).reshape(-1, 1)
        >>> name, forecast, rmse = get_best_baseline(train, test)
        >>> name in ['persistence', 'drift', 'mean', 'seasonal_naive']
        True
    """
    horizon = len(test)
    baselines = {}

    # Always try these
    baselines["persistence"] = persistence(train, horizon)
    baselines["mean"] = mean_forecast(train, horizon)

    # Drift requires at least 2 points
    if len(train) >= 2:
        baselines["drift"] = drift(train, horizon)

    # Seasonal naive requires period
    if period is not None and len(train) >= period:
        baselines["seasonal_naive"] = seasonal_naive(train, horizon, period)

    # Evaluate RMSE for each
    best_name = None
    best_forecast = None
    best_rmse = float("inf")

    for name, forecast in baselines.items():
        error = test - forecast
        rmse = float(np.sqrt(np.mean(error**2)))

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_forecast = forecast

    assert best_name is not None and best_forecast is not None
    return best_name, best_forecast, best_rmse
