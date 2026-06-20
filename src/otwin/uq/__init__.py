"""Uncertainty quantification for digital twins.

Core (numpy-only):
    - ensemble.py: ensemble forecasts, mean/std and quantile prediction intervals
    - calibration.py: PIT, coverage curves, calibration error, interval score,
      recalibration

Optional extra (``[gp]``):
    - gp_phs.py: GP residual surrogate with calibrated predictive variance
"""

from otwin.uq.calibration import (
    coverage_curve,
    expected_calibration_error,
    interval_score,
    pit_values,
    recalibrate,
    sharpness,
)
from otwin.uq.ensemble import Ensemble

__all__ = [
    "Ensemble",
    "pit_values",
    "coverage_curve",
    "expected_calibration_error",
    "interval_score",
    "recalibrate",
    "sharpness",
]
