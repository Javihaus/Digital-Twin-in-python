"""Tests for the learned PHNN. Requires the [torch] extra; skips cleanly without."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from otwin.learn import PortHamiltonianNN  # noqa: E402


def test_structure_holds_by_construction() -> None:
    model = PortHamiltonianNN(n_states=2, n_inputs=1, hidden=16, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal(2)
        s = model.check_structure(x)
        assert s["J_skew"][0], f"J not skew: {s['J_skew']}"
        assert s["R_psd"][0], f"R not PSD: {s['R_psd']}"


def test_energy_non_increasing_with_zero_input() -> None:
    model = PortHamiltonianNN(n_states=2, n_inputs=0, hidden=16, seed=1)
    from otwin.integrate import integrate_phs

    t = np.linspace(0, 5, 100)
    res = integrate_phs(model, np.array([1.0, -0.5]), t)
    assert res["success"]
    energies = np.array([model.energy(x) for x in res["x"]])
    # Passivity by construction: energy should not grow (allow tiny numerical slack).
    assert energies[-1] <= energies[0] + 1e-6


def test_fit_reduces_loss() -> None:
    # Target: a simple linear damped field dx = -0.5 x (autonomous).
    rng = np.random.default_rng(2)
    X = rng.standard_normal((128, 2))
    dXdt = -0.5 * X
    model = PortHamiltonianNN(n_states=2, n_inputs=0, hidden=16, seed=3)
    hist = model.fit(X, dXdt, epochs=150, lr=1e-2)
    assert hist["history"][-1] < hist["history"][0]
