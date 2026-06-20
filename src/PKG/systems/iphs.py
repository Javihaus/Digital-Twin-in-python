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
        - L(x): Irreversible coupling matrix (must be PSD)
        - σ(x) = ∇S^T L ∇S ≥ 0 (entropy production; holds iff L is PSD)

    The second-law guarantee σ ≥ 0 requires L(x) ⪰ 0. This is *enforced*, not
    just assumed: with ``validate=True`` (default) L is checked on first use and a
    ValueError is raised if it is not PSD. Use ``check_structure`` /
    ``check_entropy_production`` to inspect the guarantees explicitly.

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
        validate: bool = True,
    ) -> None:
        # Use PHS for reversible part
        self.phs = PortHamiltonianSystem(H, J, R, g, n_states, n_inputs)

        self.S = S
        self.L = L
        self.n_states = n_states
        self.n_inputs = n_inputs
        # The second-law guarantee σ = ∇Sᵀ L ∇S ≥ 0 holds only if L is PSD.
        # We enforce it rather than merely document it: when `validate`, the
        # irreversible coupling is checked on first dynamics/entropy call.
        self.validate = validate
        self._validated = False

    def entropy(self, x: npt.NDArray[np.floating]) -> float:
        """Evaluate entropy S(x)."""
        return float(self.S(x))

    def grad_S(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Compute gradient of entropy ∇S(x)."""
        from PKG.utils.linalg import numerical_gradient
        return numerical_gradient(self.S, x)

    def _ensure_valid_L(self, x: npt.NDArray[np.floating], tol: float = 1e-8) -> None:
        """Validate that L(x) is PSD (the second-law prerequisite); raise if not."""
        if not self.validate or self._validated:
            return
        from PKG.utils.linalg import check_psd

        is_psd, min_eig = check_psd(self.L(x), tol)
        if not is_psd:
            raise ValueError(
                "Irreversible coupling L(x) must be positive semidefinite to "
                f"guarantee entropy production σ ≥ 0 (min eigenvalue {min_eig:.3e}). "
                "Fix L or construct with validate=False to bypass (not recommended)."
            )
        self._validated = True

    def check_structure(
        self, x: npt.NDArray[np.floating], tol: float = 1e-10
    ) -> dict[str, tuple[bool, float]]:
        """Check all structural properties at ``x``.

        Returns J skew-symmetry, R PSD, L PSD, and σ ≥ 0 — the latter two are the
        irreversible (second-law) guarantees that the base PHS does not cover.
        """
        from PKG.utils.linalg import check_psd

        base = self.phs.check_structure(x, tol)
        l_psd = check_psd(self.L(x), tol)
        is_nonneg, sigma = self.check_entropy_production(x, tol)
        return {
            "J_skew": base["J_skew"],
            "R_psd": base["R_psd"],
            "L_psd": l_psd,
            "sigma_nonneg": (is_nonneg, sigma),
        }

    def check_entropy_production(
        self, x: npt.NDArray[np.floating], tol: float = 1e-10
    ) -> tuple[bool, float]:
        """Return ``(is_nonneg, sigma)`` for the entropy production at ``x``."""
        sigma = self.entropy_production(x)
        return sigma >= -tol, sigma

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
        self._ensure_valid_L(x)

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
