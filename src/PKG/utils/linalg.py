"""Linear algebra utilities for port-Hamiltonian systems."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def skew_symmetric(A: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Enforce skew-symmetry on a matrix: J = (A - A^T) / 2.

    Args:
        A: Input matrix (n x n)

    Returns:
        Skew-symmetric matrix satisfying J = -J^T

    Example:
        >>> A = np.array([[0, 2], [1, 0]])
        >>> J = skew_symmetric(A)
        >>> np.allclose(J, -J.T)
        True
    """
    return np.asarray(0.5 * (A - A.T), dtype=float)


def psd_from_cholesky(L: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Construct positive semidefinite matrix from Cholesky factor: R = L @ L^T.

    Args:
        L: Lower triangular Cholesky factor (n x n)

    Returns:
        Positive semidefinite matrix R satisfying R = R^T and R ⪰ 0

    Example:
        >>> L = np.array([[1, 0], [0.5, 0.5]])
        >>> R = psd_from_cholesky(L)
        >>> np.all(np.linalg.eigvalsh(R) >= -1e-10)
        True
    """
    return np.asarray(L @ L.T, dtype=float)


def numerical_gradient(
    f: Callable[[npt.NDArray[np.floating]], float],
    x: npt.NDArray[np.floating],
    eps: float = 1e-7,
) -> npt.NDArray[np.floating]:
    """
    Compute numerical gradient using central differences.

    Args:
        f: Scalar function R^n -> R
        x: Point at which to evaluate gradient (n,)
        eps: Finite difference step size

    Returns:
        Gradient vector (n,)

    Example:
        >>> f = lambda x: x[0]**2 + x[1]**2
        >>> x = np.array([1.0, 2.0])
        >>> grad = numerical_gradient(f, x)
        >>> np.allclose(grad, [2.0, 4.0], atol=1e-5)
        True
    """
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return grad


def check_skew_symmetric(
    J: npt.NDArray[np.floating], tol: float = 1e-10
) -> tuple[bool, float]:
    """
    Check if matrix is skew-symmetric within tolerance.

    Args:
        J: Matrix to check (n x n)
        tol: Tolerance for skew-symmetry violation

    Returns:
        (is_skew, max_violation) where max_violation = ||J + J^T||_∞

    Example:
        >>> J = np.array([[0, 1], [-1, 0]])
        >>> is_skew, viol = check_skew_symmetric(J)
        >>> is_skew
        True
    """
    violation = np.abs(J + J.T).max()
    return violation <= tol, float(violation)


def check_psd(R: npt.NDArray[np.floating], tol: float = 1e-10) -> tuple[bool, float]:
    """
    Check if matrix is positive semidefinite within tolerance.

    Args:
        R: Matrix to check (n x n)
        tol: Tolerance for negative eigenvalues

    Returns:
        (is_psd, min_eigenvalue)

    Example:
        >>> R = np.array([[1, 0], [0, 2]])
        >>> is_psd, min_eig = check_psd(R)
        >>> is_psd
        True
    """
    # Check symmetry first
    if not np.allclose(R, R.T, atol=tol):
        return False, float(-np.inf)

    eigvals = np.linalg.eigvalsh(R)
    min_eig = float(eigvals.min())
    return min_eig >= -tol, min_eig
