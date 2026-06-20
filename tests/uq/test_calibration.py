"""Tests for UQ calibration diagnostics (core, numpy-only)."""

import numpy as np

from PKG.uq import (
    coverage_curve,
    expected_calibration_error,
    interval_score,
    pit_values,
    recalibrate,
)


def _gaussian_ensemble(n=4000, m=200, seed=0):
    """A well-specified ensemble: truth and members drawn from the same N(0,1)."""
    rng = np.random.default_rng(seed)
    y_true = rng.standard_normal(n)
    ensemble = rng.standard_normal((n, m))
    return y_true, ensemble


def test_pit_is_uniform_for_calibrated_ensemble() -> None:
    y_true, ensemble = _gaussian_ensemble()
    pit = pit_values(y_true, ensemble)
    assert pit.shape == y_true.shape
    assert pit.min() >= 0.0 and pit.max() <= 1.0
    # Calibrated -> PIT ~ Uniform(0,1): mean ~ 0.5, std ~ 1/sqrt(12) ~ 0.289
    assert abs(pit.mean() - 0.5) < 0.02
    assert abs(pit.std() - (1 / np.sqrt(12))) < 0.02


def test_coverage_matches_nominal_when_calibrated() -> None:
    y_true, ensemble = _gaussian_ensemble()
    cc = coverage_curve(y_true, ensemble)
    # Empirical coverage tracks the nominal level closely.
    assert np.max(np.abs(cc["levels"] - cc["coverage"])) < 0.03


def test_expected_calibration_error_small_when_calibrated() -> None:
    y_true, ensemble = _gaussian_ensemble()
    ece = expected_calibration_error(y_true, ensemble)
    assert ece < 0.02


def test_overconfident_ensemble_undercovers() -> None:
    # Members too tight (std 0.3) around 0 while truth is N(0,1): coverage < nominal.
    rng = np.random.default_rng(1)
    y_true = rng.standard_normal(3000)
    ensemble = 0.3 * rng.standard_normal((3000, 200))
    cc = coverage_curve(y_true, ensemble, levels=np.array([0.9]))
    assert cc["coverage"][0] < 0.9
    assert expected_calibration_error(y_true, ensemble) > 0.1


def test_interval_score_rewards_calibrated_over_too_wide() -> None:
    rng = np.random.default_rng(2)
    y = rng.standard_normal(2000)
    # Calibrated-ish 90% interval for N(0,1): +-1.645
    s_good = interval_score(
        y, np.full_like(y, -1.645), np.full_like(y, 1.645), level=0.9
    )
    # Needlessly wide interval: larger (worse) score
    s_wide = interval_score(y, np.full_like(y, -5.0), np.full_like(y, 5.0), level=0.9)
    assert s_good < s_wide


def test_recalibrate_returns_monotonic_map_in_unit_interval() -> None:
    rng = np.random.default_rng(3)
    pit_cal = rng.uniform(size=500)
    R = recalibrate(pit_cal)
    levels = np.linspace(0.0, 1.0, 11)
    mapped = R(levels)
    assert mapped.min() >= 0.0 and mapped.max() <= 1.0
    assert np.all(np.diff(mapped) >= -1e-9)  # monotonic non-decreasing
