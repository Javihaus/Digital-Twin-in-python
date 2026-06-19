"""Tests for evaluation metrics with known ground truth."""

import numpy as np
import pytest

from PKG.evaluation.metrics import (
    crps,
    mae,
    mase,
    mpiw,
    nrmse,
    picp,
    rmse,
    skill_score,
    theil_u,
)


def test_rmse_perfect_prediction() -> None:
    """RMSE should be 0 for perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_pred) == 0.0


def test_rmse_known_value() -> None:
    """RMSE with known ground truth."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    # Errors: [0.1, 0.1, 0.1], RMSE = 0.1
    assert np.isclose(rmse(y_true, y_pred), 0.1)


def test_mae_known_value() -> None:
    """MAE with known ground truth."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    # Errors: [1, 1, 1], MAE = 1.0
    assert mae(y_true, y_pred) == 1.0


def test_nrmse_range_normalization() -> None:
    """nRMSE with range normalization."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
    # RMSE = 0.1, Range = 4.0, nRMSE = 0.1/4.0 = 0.025
    result = nrmse(y_true, y_pred, normalization="range")
    assert np.isclose(result, 0.025)


def test_mase_better_than_naive() -> None:
    """MASE < 1.0 means better than naive."""
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([6.0, 7.0])
    y_pred = np.array([6.05, 7.05])

    # Naive in-sample MAE = mean(|diff(train)|) = 1.0
    # Model MAE = 0.05
    # MASE = 0.05 / 1.0 = 0.05
    result = mase(y_true, y_pred, y_train)
    assert result < 1.0  # Better than naive


def test_theil_u_better_than_persistence() -> None:
    """Theil U < 1.0 means better than persistence."""
    y_train = np.array([1.0, 2.0, 3.0])
    y_true = np.array([4.0, 5.0])
    y_pred = np.array([4.1, 5.1])

    # Persistence: repeat last value (3.0)
    # Persistence RMSE = sqrt(mean([(4-3)^2, (5-3)^2])) = sqrt(2.5) ≈ 1.58
    # Model RMSE = 0.1
    # Theil U = 0.1 / 1.58 ≈ 0.06
    result = theil_u(y_true, y_pred, y_train)
    assert result < 1.0


def test_picp_perfect_coverage() -> None:
    """PICP = 1.0 when all points within bounds."""
    y_true = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.5, 1.5, 2.5])
    upper = np.array([1.5, 2.5, 3.5])
    assert picp(y_true, lower, upper) == 1.0


def test_picp_partial_coverage() -> None:
    """PICP = 0.5 when half of points within bounds."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.5, 1.5, 2.5, 5.0])  # Last one excludes true value
    upper = np.array([1.5, 2.5, 3.5, 6.0])
    assert picp(y_true, lower, upper) == 0.75  # 3 out of 4


def test_mpiw_known_value() -> None:
    """MPIW with known interval widths."""
    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([1.0, 2.0, 3.0])
    # Widths: [1.0, 1.0, 1.0], Mean = 1.0
    assert mpiw(lower, upper) == 1.0


def test_skill_score_perfect() -> None:
    """Skill score = 1.0 for perfect model (error = 0)."""
    assert skill_score(model_error=0.0, baseline_error=1.0) == 1.0


def test_skill_score_same_as_baseline() -> None:
    """Skill score = 0.0 when model same as baseline."""
    assert skill_score(model_error=0.5, baseline_error=0.5) == 0.0


def test_skill_score_worse_than_baseline() -> None:
    """Skill score < 0.0 when model worse than baseline."""
    ss = skill_score(model_error=1.0, baseline_error=0.5)
    assert ss < 0.0
    assert ss == -1.0  # 100% worse


def test_crps_deterministic() -> None:
    """CRPS for deterministic forecast (single ensemble member)."""
    y_true = np.array([1.0, 2.0])
    ensemble = np.array([[1.0], [2.0]])  # Perfect forecast

    score = crps(y_true, ensemble)
    assert score == 0.0


def test_crps_ensemble() -> None:
    """CRPS for ensemble forecast."""
    y_true = np.array([1.0, 2.0])
    ensemble = np.array([[0.9, 1.0, 1.1], [1.9, 2.0, 2.1]])

    score = crps(y_true, ensemble)
    assert score < 0.2  # Should be small for good forecast
    assert score > 0.0  # Not perfect
