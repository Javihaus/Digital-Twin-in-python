"""Irreversible Port-Hamiltonian Systems (IPHS) with entropy production."""

from typing import Callable

import numpy as np
import numpy.typing as npt

from PKG.systems.phs import PortHamiltonianSystem


class IrreversiblePHS:
    """
    Irreversible Port-Hamiltonian System with entropy production.

    Extension of PHS to include thermodynamic irreversibility:
        ẋ = (J(x) − R(x)) ∇H(x) + g(x) u + L(x) ∇S(x)
        y = g(x)^T ∇H(x)
        σ(x) ≥ 0  (entropy production, second law)

    where:
        - S(x): Entropy/availability function
        - L(x): Irreversible coupling matrix
        - σ(x) = ∇S^T L ∇S ≥ 0 (guaranteed by structure)

    This is the principled approach to dissipative systems.

    Args:
        H: Energy (internal energy or Hamiltonian)
        S: Entropy/availability function
        J: Interconnection (skew-symmetric)
        R: Dissipation (PSD)
        L: Irreversible coupling (must ensure σ ≥ 0)
        g: Input map
        n_states: Number of states
        n_inputs: Number of inputs

    Example:
        >>> # Simple 1D irreversible system
        >>> H = lambda x: 0.5 * x[0]**2
        >>> S = lambda x: -x[0]  # Entropy increases as energy decreases
        >>> J = lambda x: np.zeros((1, 1))
        >>> R = lambda x: np.array([[0.1]])
        >>> L = lambda x: np.array([[0.05]])
        >>> g = lambda x: np.array([[1.0]])
        >>> iphs = IrreversiblePHS(H, S, J, R, L, g, 1, 1)
    """

    def __init__(
        self,
        H: Callable[[npt.NDArray[np.floating]], float],
        S: Callable[[npt.NDArray[np.floating]], float],
        J: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        R: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        L: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        g: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        n_states: int,
        n_inputs: int,
    ) -> None:
        # Use PHS for reversible part
        self.phs = PortHamiltonianSystem(H, J, R, g, n_states, n_inputs)

        self.S = S
        self.L = L

    def entropy(self, x: npt.NDArray[np.floating]) -> float:
        """Evaluate entropy S(x)."""
        return float(self.S(x))

    def grad_S(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Compute gradient of entropy ∇S(x)."""
        from PKG.utils.linalg import numerical_gradient
        return numerical_gradient(self.S, x)

    def dynamics(
        self,
        x: npt.NDArray[np.floating],
        u: npt.NDArray[np.floating],
        t: float = 0.0,
    ) -> npt.NDArray[np.floating]:
        """
        Compute state derivative: ẋ = (J - R) ∇H + g u + L ∇S.

        The irreversible term L ∇S captures entropy production.
        """
        # Reversible + dissipative part
        dx_phs = self.phs.dynamics(x, u, t)

        # Irreversible part
        grad_S = self.grad_S(x)
        L_mat = self.L(x)
        dx_irreversible = L_mat @ grad_S

        return dx_phs + dx_irreversible

    def entropy_production(self, x: npt.NDArray[np.floating]) -> float:
        """
        Compute entropy production: σ = ∇S^T L ∇S.

        Must be non-negative (second law).

        Returns:
            Entropy production rate (should be ≥ 0)
        """
        grad_S = self.grad_S(x)
        L_mat = self.L(x)

        sigma = float(grad_S @ L_mat @ grad_S)
        return sigma
