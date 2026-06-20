"""Tests for ensemble forecasting (core, numpy-only)."""

import numpy as np
import pytest

from otwin.evaluation.metrics import crps
from otwin.systems import water_tank
from otwin.twin import DigitalTwin
from otwin.uq import Ensemble


def _ensemble():
    members = [DigitalTwin(water_tank(a=a)) for a in (0.08, 0.10, 0.12)]
    return Ensemble(members)


def test_requires_at_least_two_members() -> None:
    with pytest.raises(ValueError):
        Ensemble([DigitalTwin(water_tank())])


def test_trajectories_shape() -> None:
    ens = _ensemble()
    t = np.linspace(0, 5, 40)
    u = np.zeros((40, 1))
    trajs = ens.forecast_trajectories(np.array([2.0]), t, u)
    assert trajs.shape == (3, 40, 1)


def test_interval_bounds_ordered_and_contain_mean() -> None:
    ens = _ensemble()
    t = np.linspace(0, 5, 40)
    u = np.zeros((40, 1))
    band = ens.forecast_interval(np.array([2.0]), t, u, level=0.9)
    assert band["lower"].shape == (40, 1)
    assert np.all(band["lower"] <= band["upper"] + 1e-12)
    assert np.all(band["mean"] >= band["lower"] - 1e-9)
    assert np.all(band["mean"] <= band["upper"] + 1e-9)
    # Different members -> real spread somewhere along the horizon.
    assert band["std"].max() > 0.0


def test_ensemble_matrix_is_crps_compatible() -> None:
    ens = _ensemble()
    t = np.linspace(0, 5, 40)
    u = np.zeros((40, 1))
    mat = ens.ensemble_matrix(np.array([2.0]), t, u, state=0)
    assert mat.shape == (40, 3)  # (n_steps, n_members)
    # crps expects (n, n_members); a "truth" equal to the member mean -> finite.
    y_true = mat.mean(axis=1)
    score = crps(y_true, mat)
    assert np.isfinite(score) and score >= 0.0


def test_level_must_be_in_unit_interval() -> None:
    ens = _ensemble()
    t = np.linspace(0, 1, 10)
    u = np.zeros((10, 1))
    with pytest.raises(ValueError):
        ens.forecast_interval(np.array([2.0]), t, u, level=1.5)
