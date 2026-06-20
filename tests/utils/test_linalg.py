"""Tests for linear algebra utilities."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from PKG.utils.linalg import (
    check_psd,
    check_skew_symmetric,
    numerical_gradient,
    psd_from_cholesky,
    skew_symmetric,
)


def test_skew_symmetric() -> None:
    """Test skew-symmetrization of a matrix."""
    A = np.array([[0, 2], [1, 0]])
    J = skew_symmetric(A)

    # Check skew-symmetry
    assert np.allclose(J, -J.T)

    # Expected result: [[0, 0.5], [-0.5, 0]]
    expected = np.array([[0, 0.5], [-0.5, 0]])
    assert np.allclose(J, expected)


def test_psd_from_cholesky() -> None:
    """Test PSD matrix construction from Cholesky factor."""
    L = np.array([[1, 0], [0.5, 0.5]])
    R = psd_from_cholesky(L)

    # Check symmetry
    assert np.allclose(R, R.T)

    # Check positive semidefinite
    eigvals = np.linalg.eigvalsh(R)
    assert np.all(eigvals >= -1e-10)


def test_numerical_gradient() -> None:
    """Test numerical gradient computation."""
    # Quadratic function: f(x) = x1^2 + 2*x2^2
    f = lambda x: x[0] ** 2 + 2 * x[1] ** 2
    x = np.array([1.0, 2.0])

    grad = numerical_gradient(f, x)

    # Expected gradient: [2*x1, 4*x2] = [2, 8]
    expected = np.array([2.0, 8.0])
    assert np.allclose(grad, expected, atol=1e-5)


def test_check_skew_symmetric() -> None:
    """Test skew-symmetry checking."""
    # Exactly skew-symmetric
    J = np.array([[0, 1], [-1, 0]])
    is_skew, violation = check_skew_symmetric(J)
    assert is_skew
    assert violation < 1e-10

    # Not skew-symmetric
    A = np.array([[1, 2], [3, 4]])
    is_skew, violation = check_skew_symmetric(A)
    assert not is_skew
    assert violation > 0.1


def test_check_psd() -> None:
    """Test PSD checking."""
    # Positive definite
    R = np.array([[2, 1], [1, 2]])
    is_psd, min_eig = check_psd(R)
    assert is_psd
    assert min_eig > 0

    # Not symmetric
    A = np.array([[1, 2], [3, 4]])
    is_psd, min_eig = check_psd(A)
    assert not is_psd

    # Negative eigenvalue
    R_neg = np.array([[1, 2], [2, 1]])
    eigvals = np.linalg.eigvalsh(R_neg)
    if eigvals.min() < 0:
        is_psd, min_eig = check_psd(R_neg)
        assert not is_psd


@given(
    n=st.integers(min_value=2, max_value=5),
)
def test_skew_symmetric_property(n: int) -> None:
    """Property: skew_symmetric always produces skew matrix."""
    A = np.random.randn(n, n)
    J = skew_symmetric(A)

    is_skew, violation = check_skew_symmetric(J)
    assert is_skew, f"Skew-symmetry violated: {violation}"


@given(
    n=st.integers(min_value=2, max_value=5),
)
def test_psd_from_cholesky_property(n: int) -> None:
    """Property: psd_from_cholesky always produces PSD matrix."""
    L = np.random.randn(n, n)
    R = psd_from_cholesky(L)

    is_psd, min_eig = check_psd(R)
    assert is_psd, f"Not PSD, min eigenvalue: {min_eig}"
