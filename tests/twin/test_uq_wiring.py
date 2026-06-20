"""Tests that UQ is wired into the twin and that fake uncertainty is banned."""

import numpy as np
import pytest

from PKG.systems import water_tank
from PKG.twin import DigitalTwin
from PKG.uq import Ensemble


def test_return_uncertainty_without_uq_raises() -> None:
    """The v1 sin: returning zero-width bands. Must raise instead."""
    twin = DigitalTwin(model=water_tank())
    t = np.linspace(0, 5, 40)
    u = np.zeros((40, 1))
    with pytest.raises(ValueError):
        twin.forecast(np.array([2.0]), t, u, return_uncertainty=True)


def test_uq_ensemble_requires_ensemble_argument() -> None:
    with pytest.raises(ValueError):
        DigitalTwin(model=water_tank(), uq="ensemble")


def test_forecast_returns_success_key() -> None:
    twin = DigitalTwin(model=water_tank())
    t = np.linspace(0, 5, 40)
    u = np.zeros((40, 1))
    out = twin.forecast(np.array([2.0]), t, u)
    assert out["success"] is True
    assert out["x"].shape == (40, 1)


def test_ensemble_uq_produces_nonzero_bands() -> None:
    members = [DigitalTwin(water_tank(a=a)) for a in (0.08, 0.10, 0.12)]
    twin = DigitalTwin(model=water_tank(), uq="ensemble", ensemble=Ensemble(members))
    t = np.linspace(0, 5, 40)
    u = np.zeros((40, 1))
    out = twin.forecast(np.array([2.0]), t, u, return_uncertainty=True, level=0.9)
    assert "lower" in out and "upper" in out
    assert np.all(out["lower"] <= out["upper"] + 1e-12)
    assert out["std"].max() > 0.0  # genuine, nonzero uncertainty


def test_predict_dt_is_explicit() -> None:
    twin = DigitalTwin(model=water_tank())
    X = np.array([[2.0], [1.0]])
    p1 = twin.predict(X, dt=0.5)
    p2 = twin.predict(X, dt=1.0)
    assert not np.allclose(p1, p2)  # dt actually affects the step


def test_assimilate_is_a_real_kalman_update() -> None:
    twin = DigitalTwin(model=water_tank())
    x_prior = np.array([2.0])
    obs = np.array([1.0])
    # Very precise observation -> posterior close to obs; very precise prior ->
    # posterior close to prior. A fixed 50/50 average could not do both.
    post_trust_obs = twin.assimilate(x_prior, obs, obs_noise=1e-3, prior_noise=1.0)["x"]
    post_trust_prior = twin.assimilate(x_prior, obs, obs_noise=1.0, prior_noise=1e-3)["x"]
    assert abs(post_trust_obs[0] - 1.0) < 1e-2
    assert abs(post_trust_prior[0] - 2.0) < 1e-2
