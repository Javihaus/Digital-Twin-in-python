"""Port-Hamiltonian System (PHS) implementation."""

from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from PKG.utils.linalg import check_psd, check_skew_symmetric, numerical_gradient


class PortHamiltonianSystem:
    """
    Port-Hamiltonian System in input-state-output form.

    Structure:
        ẋ = (J(x) - R(x)) ∇H(x) + g(x) u
        y = g(x)^T ∇H(x)

    where:
        - J(x) = -J(x)^T  (skew-symmetric, lossless interconnection)
        - R(x) ⪰ 0        (positive semidefinite, dissipation)
        - H(x)            (energy/storage function)
        - g(x)            (input/output map)

    Power balance (enforced structurally):
        dH/dt = ∇H^T ẋ = -∇H^T R ∇H + y^T u ≤ y^T u

    With u=0, energy is non-increasing: dH/dt ≤ 0 (passivity by construction).

    Args:
        H: Energy function H(x) -> scalar
        J: Interconnection matrix J(x) -> (n, n), must be skew-symmetric
        R: Dissipation matrix R(x) -> (n, n), must be PSD
        g: Input map g(x) -> (n, m)
        n_states: Number of states
        n_inputs: Number of inputs
        grad_H: Optional gradient function (if None, uses numerical gradient)

    Example:
        >>> # Simple 1D dissipative system
        >>> H = lambda x: 0.5 * x[0]**2
        >>> J = lambda x: np.zeros((1, 1))
        >>> R = lambda x: np.array([[0.1]])
        >>> g = lambda x: np.array([[1.0]])
        >>> phs = PortHamiltonianSystem(H, J, R, g, n_states=1, n_inputs=1)
        >>> x = np.array([1.0])
        >>> u = np.array([0.0])
        >>> dx = phs.dynamics(x, u)
        >>> dx[0] < 0  # Energy decreases
        True
    """

    def __init__(
        self,
        H: Callable[[npt.NDArray[np.floating]], float],
        J: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        R: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        g: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        n_states: int,
        n_inputs: int,
        grad_H: Optional[
            Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
        ] = None,
    ) -> None:
        self.H = H
        self.J = J
        self.R = R
        self.g = g
        self.n_states = n_states
        self.n_inputs = n_inputs
        self._grad_H = grad_H

    def energy(self, x: npt.NDArray[np.floating]) -> float:
        """Evaluate energy H(x)."""
        return float(self.H(x))

    def grad_H(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Compute gradient of energy ∇H(x)."""
        if self._grad_H is not None:
            return self._grad_H(x)
        return numerical_gradient(self.H, x)

    def dynamics(
        self,
        x: npt.NDArray[np.floating],
        u: npt.NDArray[np.floating],
        t: float = 0.0,
    ) -> npt.NDArray[np.floating]:
        """
        Compute state derivative: ẋ = (J - R) ∇H + g u.

        Args:
            x: State (n_states,)
            u: Input (n_inputs,)
            t: Time (unused, for compatibility with integrators)

        Returns:
            State derivative ẋ (n_states,)
        """
        grad_H = self.grad_H(x)
        J_mat = self.J(x)
        R_mat = self.R(x)
        g_mat = self.g(x)

        dx = (J_mat - R_mat) @ grad_H + g_mat @ u
        return dx

    def output(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Compute output: y = g^T ∇H.

        Args:
            x: State (n_states,)

        Returns:
            Output y (n_inputs,)
        """
        grad_H = self.grad_H(x)
        g_mat = self.g(x)
        return g_mat.T @ grad_H

    def power_balance(
        self, x: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> dict[str, float]:
        """
        Compute power balance terms.

        Returns:
            Dictionary with:
                - 'dH_dt': Time derivative of energy
                - 'dissipated': Power dissipated (-∇H^T R ∇H ≤ 0)
                - 'supplied': Power supplied (y^T u)

        The power balance identity must hold:
            dH_dt = dissipated + supplied
        """
        grad_H = self.grad_H(x)
        R_mat = self.R(x)
        y = self.output(x)
        dx = self.dynamics(x, u)

        dH_dt = float(grad_H @ dx)
        dissipated = float(-grad_H @ R_mat @ grad_H)
        supplied = float(y @ u)

        return {
            "dH_dt": dH_dt,
            "dissipated": dissipated,
            "supplied": supplied,
        }

    def check_structure(
        self, x: npt.NDArray[np.floating], tol: float = 1e-10
    ) -> dict[str, tuple[bool, float]]:
        """
        Check structural properties at state x.

        Args:
            x: State to check
            tol: Numerical tolerance

        Returns:
            Dictionary with:
                - 'J_skew': (is_skew, max_violation)
                - 'R_psd': (is_psd, min_eigenvalue)
        """
        J_mat = self.J(x)
        R_mat = self.R(x)

        return {
            "J_skew": check_skew_symmetric(J_mat, tol),
            "R_psd": check_psd(R_mat, tol),
        }
