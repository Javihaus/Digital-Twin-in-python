"""Tests for the structure-preserving (implicit-midpoint) integrator."""

import numpy as np

from PKG.integrate import implicit_midpoint, integrate_phs, integrate_with_inputs
from PKG.systems import water_tank


def test_energy_is_monotonically_non_increasing() -> None:
    """The whole point: no spurious energy injection over a long horizon."""
    tank = water_tank()
    t = np.linspace(0, 50, 500)
    u = np.zeros((500, 1))
    res = integrate_phs(tank, np.array([3.0]), t, u)
    assert res["success"]
    energies = np.array([tank.energy(x) for x in res["x"]])
    diffs = np.diff(energies)
    assert np.all(diffs <= 1e-9), f"energy increased: max diff {diffs.max():.2e}"


def test_dispatch_via_integrate_with_inputs() -> None:
    tank = water_tank()
    t = np.linspace(0, 10, 100)
    u = np.zeros((100, 1))
    res = integrate_with_inputs(
        lambda tv, x, uv: tank.dynamics(x, uv),
        np.array([2.0]),
        t,
        u,
        method="implicit_midpoint",
    )
    assert res["success"]
    assert res["x"].shape == (100, 1)


def test_matches_analytic_linear_decay() -> None:
    """dx/dt = -x has solution x(t) = x0 e^{-t}; midpoint should be close."""
    t = np.linspace(0, 2, 200)
    u = np.zeros((200, 1))
    res = implicit_midpoint(lambda tv, x, uv: -x, np.array([1.0]), t, u)
    assert res["success"]
    analytic = np.exp(-t)
    assert np.max(np.abs(res["x"][:, 0] - analytic)) < 1e-3


def test_rejects_non_increasing_time() -> None:
    import pytest

    with pytest.raises(ValueError):
        implicit_midpoint(
            lambda tv, x, uv: -x, np.array([1.0]), np.array([0.0, 0.0, 1.0]), np.zeros((3, 1))
        )
