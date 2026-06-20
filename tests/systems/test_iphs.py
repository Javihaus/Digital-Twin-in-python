"""Tests for Irreversible PHS with entropy production."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from otwin.systems.iphs import IrreversiblePHS


@given(
    x_val=st.floats(min_value=0.1, max_value=5.0),
)
@settings(deadline=500)
def test_entropy_production_nonnegative(x_val: float) -> None:
    """Property: Entropy production σ ≥ 0 (second law)."""
    # Simple 1D IPHS
    H = lambda x: 0.5 * x[0] ** 2
    S = lambda x: -x[0]  # Entropy increases as energy decreases
    J = lambda x: np.zeros((1, 1))
    R = lambda x: np.array([[0.1]])

    # L must be PSD to ensure σ ≥ 0
    L = lambda x: np.array([[0.05]])  # Positive constant
    g = lambda x: np.array([[1.0]])

    iphs = IrreversiblePHS(H, S, J, R, L, g, n_states=1, n_inputs=1)

    x = np.array([x_val])
    sigma = iphs.entropy_production(x)

    # Second law: entropy production must be non-negative
    assert sigma >= -1e-10, f"Entropy production negative: σ = {sigma}"


def test_iphs_reduces_to_phs_when_L_zero() -> None:
    """When L = 0, IPHS reduces to PHS."""
    H = lambda x: 0.5 * x[0] ** 2
    S = lambda x: 0.0  # Dummy entropy
    J = lambda x: np.zeros((1, 1))
    R = lambda x: np.array([[0.1]])
    L = lambda x: np.zeros((1, 1))  # No irreversible coupling
    g = lambda x: np.array([[1.0]])

    iphs = IrreversiblePHS(H, S, J, R, L, g, 1, 1)

    x = np.array([1.0])
    u = np.array([0.0])

    # Should match PHS dynamics
    dx_iphs = iphs.dynamics(x, u)
    dx_phs = iphs.phs.dynamics(x, u)

    assert np.allclose(dx_iphs, dx_phs)


def test_entropy_production_zero_at_equilibrium() -> None:
    """At equilibrium (∇S = 0), entropy production is zero."""
    H = lambda x: 0.5 * x[0] ** 2
    S = lambda x: x[0] ** 2 - 1.0  # Minimum at x = 0 (∇S = 0 there)
    J = lambda x: np.zeros((1, 1))
    R = lambda x: np.array([[0.1]])
    L = lambda x: np.array([[0.05]])
    g = lambda x: np.array([[1.0]])

    iphs = IrreversiblePHS(H, S, J, R, L, g, 1, 1)

    x_eq = np.array([0.0])  # Equilibrium where ∇S = 0
    sigma = iphs.entropy_production(x_eq)

    assert abs(sigma) < 1e-6, f"Entropy production at equilibrium: {sigma}"
