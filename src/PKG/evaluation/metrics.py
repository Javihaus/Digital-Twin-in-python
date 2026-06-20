"""Evaluation metrics for forecasting (calibrated, scale-free)."""

import numpy as np
import numpy.typing as npt


def rmse(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: True values (n,) or (n, d)
        y_pred: Predicted values (n,) or (n, d)

    Returns:
        RMSE (lower is better)

    Example:
        >>> y_true = np.array([1, 2, 3])
        >>> y_pred = np.array([1.1, 2.1, 2.9])
        >>> rmse(y_true, y_pred)
        0.1
    """
    error = y_true - y_pred
    return float(np.sqrt(np.mean(error**2)))


def mae(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE (lower is better)
    """
    error = np.abs(y_true - y_pred)
    return float(np.mean(error))


def nrmse(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    normalization: str = "range",
) -> float:
    """
    Normalized Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values
        normalization: 'range' (max - min) or 'mean' (mean of y_true)

    Returns:
        nRMSE (dimensionless, lower is better)

    Example:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
        >>> nrmse(y_true, y_pred, normalization='range')
        0.025
    """
    error_rmse = rmse(y_true, y_pred)

    if normalization == "range":
        norm = float(y_true.max() - y_true.min())
    elif normalization == "mean":
        norm = float(np.mean(y_true))
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if norm == 0:
        return float("inf")

    return error_rmse / norm


def mase(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.floating],
    seasonal_period: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler).

    Scale-free metric: compares model MAE to in-sample naive forecast MAE.
    MASE < 1.0 means better than naive; MASE > 1.0 means worse.

    Args:
        y_true: True test values
        y_pred: Model predictions
        y_train: Training data (for computing naive forecast error)
        seasonal_period: Period for seasonal naive (default: 1 for persistence)

    Returns:
        MASE (< 1.0 is better than naive)

    Example:
        >>> y_train = np.array([1, 2, 3, 4, 5])
        >>> y_true = np.array([6, 7])
        >>> y_pred = np.array([6.1, 7.1])
        >>> mase(y_true, y_pred, y_train)
        0.1
    """
    # Model error
    model_mae = mae(y_true, y_pred)

    # Naive in-sample error (persistence or seasonal naive)
    if seasonal_period == 1:
        # Persistence: y[t] predicted by y[t-1]
        naive_errors = np.abs(np.diff(y_train.flatten()))
    else:
        # Seasonal naive: y[t] predicted by y[t-period]
        naive_errors = np.abs(
            y_train.flatten()[seasonal_period:] - y_train.flatten()[:-seasonal_period]
        )

    naive_mae = float(np.mean(naive_errors))

    if naive_mae == 0:
        return float("inf") if model_mae > 0 else 0.0

    return model_mae / naive_mae


def theil_u(
    y_true: npt.NDArray[np.floating],
    y_pred: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.floating] | None = None,
) -> float:
    """
    Theil's U statistic (forecast accuracy relative to naive forecast).

    U < 1.0 means better than persistence; U > 1.0 means worse.

    Args:
        y_true: True test values (n,) or (n, d)
        y_pred: Model predictions
        y_train: Training data (if None, uses last test value for persistence)

    Returns:
        Theil U (< 1.0 is better than naive)

    Example:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
        >>> theil_u(y_true, y_pred)
        0.1
    """
    # Model RMSE
    model_rmse = rmse(y_true, y_pred)

    # Naive forecast: persistence (last known value)
    if y_train is not None:
        last_value = y_train[-1]
    else:
        # Fall back to first test value
        last_value = y_true[0]

    naive_pred = np.full_like(y_true, last_value)
    naive_rmse = rmse(y_true, naive_pred)

    if naive_rmse == 0:
        return float("inf") if model_rmse > 0 else 0.0

    return model_rmse / naive_rmse


def crps(
    y_true: npt.NDArray[np.floating],
    ensemble_forecasts: npt.NDArray[np.floating],
) -> float:
    """
    Continuous Ranked Probability Score (probabilistic metric).

    Measures calibration + sharpness of ensemble forecasts.
    Lower is better.

    Args:
        y_true: True values (n,)
        ensemble_forecasts: Ensemble predictions (n, n_members)

    Returns:
        CRPS (lower is better)

    Example:
        >>> y_true = np.array([1.0, 2.0])
        >>> ensemble = np.array([[0.9, 1.0, 1.1], [1.9, 2.0, 2.1]])
        >>> score = crps(y_true, ensemble)
        >>> score < 0.2
        True
    """
    if ensemble_forecasts.ndim == 1:
        ensemble_forecasts = ensemble_forecasts.reshape(-1, 1)

    n = len(y_true)
    crps_values = []

    for i in range(n):
        # Empirical CDF from ensemble
        ensemble = ensemble_forecasts[i]
        true_val = y_true[i]

        # CRPS = E|X - y| - 0.5 E|X - X'|
        # where X, X' are independent samples from forecast distribution
        term1 = np.mean(np.abs(ensemble - true_val))
        term2 = 0.5 * np.mean(
            np.abs(ensemble[:, None] - ensemble[None, :])
        )  # All pairs

        crps_values.append(term1 - term2)

    return float(np.mean(crps_values))


def picp(
    y_true: npt.NDArray[np.floating],
    lower: npt.NDArray[np.floating],
    upper: npt.NDArray[np.floating],
) -> float:
    """
    Prediction Interval Coverage Probability.

    Fraction of true values within [lower, upper] bounds.
    Should be close to nominal level (e.g., 0.90 for 90% intervals).

    Args:
        y_true: True values (n,)
        lower: Lower bounds (n,)
        upper: Upper bounds (n,)

    Returns:
        Coverage fraction in [0, 1]

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> lower = np.array([0.5, 1.5, 2.5])
        >>> upper = np.array([1.5, 2.5, 3.5])
        >>> picp(y_true, lower, upper)
        1.0
    """
    within_bounds = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within_bounds))


def mpiw(
    lower: npt.NDArray[np.floating],
    upper: npt.NDArray[np.floating],
) -> float:
    """
    Mean Prediction Interval Width.

    Average width of prediction intervals (sharpness metric).
    Lower is better (sharper), but must be balanced with coverage (PICP).

    Args:
        lower: Lower bounds (n,)
        upper: Upper bounds (n,)

    Returns:
        Mean interval width

    Example:
        >>> lower = np.array([0.5, 1.5, 2.5])
        >>> upper = np.array([1.5, 2.5, 3.5])
        >>> mpiw(lower, upper)
        1.0
    """
    widths = upper - lower
    return float(np.mean(widths))


def skill_score(
    model_error: float,
    baseline_error: float,
) -> float:
    """
    Skill score: (baseline_error - model_error) / baseline_error.

    Positive score means model is better than baseline.
    Score of 0.5 means 50% error reduction vs baseline.

    Args:
        model_error: Model error (RMSE, MAE, etc.)
        baseline_error: Baseline error (same metric)

    Returns:
        Skill score (higher is better, negative means worse than baseline)

    Example:
        >>> skill_score(model_error=0.1, baseline_error=0.2)
        0.5
    """
    if baseline_error == 0:
        return 0.0 if model_error == 0 else float("-inf")

    return (baseline_error - model_error) / baseline_error
