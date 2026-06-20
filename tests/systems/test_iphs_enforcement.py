"""Tests that IPHS enforces the second-law prerequisite (L PSD, sigma >= 0)."""

import numpy as np
import pytest

from otwin.systems.iphs import IrreversiblePHS


def _valid_iphs(validate=True):
    H = lambda x: 0.5 * x[0] ** 2
    S = lambda x: -x[0]
    J = lambda x: np.zeros((1, 1))
    R = lambda x: np.array([[0.1]])
    L = lambda x: np.array([[0.05]])  # PSD
    g = lambda x: np.array([[1.0]])
    return IrreversiblePHS(H, S, J, R, L, g, 1, 1, validate=validate)


def test_entropy_production_nonnegative_for_valid_L() -> None:
    iphs = _valid_iphs()
    for h in np.linspace(0.1, 3.0, 10):
        is_nonneg, sigma = iphs.check_entropy_production(np.array([h]))
        assert is_nonneg and sigma >= -1e-10


def test_check_structure_reports_all_guarantees() -> None:
    iphs = _valid_iphs()
    s = iphs.check_structure(np.array([1.0]))
    assert s["J_skew"][0]
    assert s["R_psd"][0]
    assert s["L_psd"][0]
    assert s["sigma_nonneg"][0]


def test_non_psd_L_is_rejected_on_use() -> None:
    H = lambda x: 0.5 * (x[0] ** 2 + x[1] ** 2)
    S = lambda x: -(x[0] + x[1])
    J = lambda x: np.zeros((2, 2))
    R = lambda x: np.eye(2) * 0.1
    # Indefinite L: eigenvalues +1 and -1 -> not PSD -> must be rejected.
    L = lambda x: np.array([[0.0, 1.0], [1.0, 0.0]])
    g = lambda x: np.eye(2)
    iphs = IrreversiblePHS(H, S, J, R, L, g, 2, 2, validate=True)
    with pytest.raises(ValueError):
        iphs.dynamics(np.array([1.0, 1.0]), np.zeros(2))


def test_validate_false_bypasses_check() -> None:
    H = lambda x: 0.5 * (x[0] ** 2 + x[1] ** 2)
    S = lambda x: -(x[0] + x[1])
    J = lambda x: np.zeros((2, 2))
    R = lambda x: np.eye(2) * 0.1
    L = lambda x: np.array([[0.0, 1.0], [1.0, 0.0]])
    g = lambda x: np.eye(2)
    iphs = IrreversiblePHS(H, S, J, R, L, g, 2, 2, validate=False)
    # Does not raise (bypassed); returns a finite derivative.
    dx = iphs.dynamics(np.array([1.0, 1.0]), np.zeros(2))
    assert np.all(np.isfinite(dx))
