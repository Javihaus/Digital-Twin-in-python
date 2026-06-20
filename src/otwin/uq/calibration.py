"""Calibration diagnostics for probabilistic forecasts.

Point metrics (RMSE, MAE) say nothing about whether the *uncertainty* is well-calibrated.
These functions test that: does a stated 90% interval actually contain the truth
~90% of the time? Are the predictive CDFs uniform under the PIT? They reuse the
interval metrics in :mod:`otwin.evaluation.metrics` rather than reimplementing them.

All functions are core (numpy only).
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from otwin.evaluation.metrics import mpiw, picp


def pit_values(
    y_true: npt.NDArray[np.floating],
    ensemble: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Probability Integral Transform values for an ensemble forecast.

    For each observation, the PIT is the fraction of ensemble members at or below
    the observed value (the empirical predictive CDF evaluated at ``y_true``). A
    well-calibrated forecaster yields PIT values that are ~Uniform(0, 1).

    Args:
        y_true: Observations ``(n,)``.
        ensemble: Member forecasts ``(n, n_members)``.

    Returns:
        PIT values ``(n,)`` in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=float)
    ensemble = np.asarray(ensemble, dtype=float)
    if ensemble.ndim != 2 or ensemble.shape[0] != y_true.shape[0]:
        raise ValueError("ensemble must have shape (n, n_members) matching y_true")
    return np.asarray((ensemble <= y_true[:, None]).mean(axis=1), dtype=float)


def coverage_curve(
    y_true: npt.NDArray[np.floating],
    ensemble: npt.NDArray[np.floating],
    levels: npt.NDArray[np.floating] | None = None,
) -> dict[str, npt.NDArray[np.floating]]:
    """Empirical coverage of central quantile intervals across nominal levels.

    Args:
        y_true: Observations ``(n,)``.
        ensemble: Member forecasts ``(n, n_members)``.
        levels: Nominal central levels in (0, 1). Defaults to 0.1..0.9.

    Returns:
        Dict with 'levels' (nominal) and 'coverage' (empirical PICP at each level).
    """
    if levels is None:
        levels = np.linspace(0.1, 0.9, 9)
    levels = np.asarray(levels, dtype=float)
    ensemble = np.asarray(ensemble, dtype=float)

    cov = np.empty_like(levels)
    for i, lvl in enumerate(levels):
        alpha = 1.0 - lvl
        lo = np.quantile(ensemble, alpha / 2.0, axis=1)
        hi = np.quantile(ensemble, 1.0 - alpha / 2.0, axis=1)
        cov[i] = picp(y_true, lo, hi)
    return {"levels": levels, "coverage": cov}


def expected_calibration_error(
    y_true: npt.NDArray[np.floating],
    ensemble: npt.NDArray[np.floating],
    levels: npt.NDArray[np.floating] | None = None,
) -> float:
    """Mean absolute gap between nominal and empirical coverage over ``levels``.

    Zero means perfectly calibrated intervals (on this sample). Lower is better.
    """
    cc = coverage_curve(y_true, ensemble, levels)
    return float(np.mean(np.abs(cc["levels"] - cc["coverage"])))


def interval_score(
    y_true: npt.NDArray[np.floating],
    lower: npt.NDArray[np.floating],
    upper: npt.NDArray[np.floating],
    level: float = 0.9,
) -> float:
    """Gneiting–Raftery interval score for central ``level`` intervals (lower=better).

    Penalises width plus a calibration penalty for observations falling outside
    the interval. Rewards sharp intervals only when they stay calibrated.

    Args:
        y_true: Observations ``(n,)``.
        lower, upper: Interval bounds ``(n,)``.
        level: Nominal central coverage in (0, 1).
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    alpha = 1.0 - level
    width = upper - lower
    below = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    above = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return float(np.mean(width + below + above))


def recalibrate(
    pit_cal: npt.NDArray[np.floating],
) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
    """Build a monotonic recalibration map from calibration-set PIT values.

    Returns a function ``R`` mapping a *nominal* quantile level ``p`` to the
    *adjusted* level whose empirical coverage matches ``p`` (Kuleshov et al.,
    2018, "Accurate Uncertainties for Deep Learning Using Calibrated Regression").
    ``R`` is the empirical CDF of the calibration PIT values, applied via
    interpolation.

    Args:
        pit_cal: PIT values on a held-out calibration set ``(n,)``.

    Returns:
        Vectorised callable mapping levels in [0, 1] to recalibrated levels.
    """
    pit_sorted = np.sort(np.asarray(pit_cal, dtype=float))
    n = pit_sorted.size
    if n == 0:
        raise ValueError("pit_cal must be non-empty")
    ecdf_y = (np.arange(1, n + 1)) / n

    def _map(p: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        return np.asarray(
            np.interp(p_arr, pit_sorted, ecdf_y, left=0.0, right=1.0), dtype=float
        )

    return _map


def sharpness(
    lower: npt.NDArray[np.floating],
    upper: npt.NDArray[np.floating],
) -> float:
    """Mean prediction interval width (alias of :func:`otwin.evaluation.metrics.mpiw`)."""
    return mpiw(lower, upper)
