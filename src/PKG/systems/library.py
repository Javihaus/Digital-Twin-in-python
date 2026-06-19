"""Library of reference port-Hamiltonian systems."""

import numpy as np
import numpy.typing as npt

from PKG.systems.phs import PortHamiltonianSystem


def water_tank(
    A: float = 1.0,
    a: float = 0.1,
    g: float = 9.81,
    c_d: float = 0.6,
    rho: float = 1000.0,
) -> PortHamiltonianSystem:
    """
    Water tank with drain (passive dissipative PHS).

    State: x = [h] (water height, m)
    Input: u = [q_in] (inflow rate, m³/s)
    Energy: H(h) = (1/2) ρ g A h² (potential energy)

    Dynamics:
        ẋ = -c_d a √(2gh) / A + u / A
          = (J - R) ∇H + g u

    where:
        J = 0 (no internal interconnection for 1D)
        R = c_d a / (ρ g A²) (dissipation from drain)
        g = 1/A (input map)
        ∇H = ρ g A h

    With u=0, energy strictly decreases (passive system).

    Args:
        A: Tank cross-sectional area (m²)
        a: Drain orifice area (m²)
        g: Gravitational acceleration (m/s²)
        c_d: Discharge coefficient (dimensionless, typically ~0.6)
        rho: Water density (kg/m³)

    Returns:
        PortHamiltonianSystem for water tank

    Example:
        >>> tank = water_tank()
        >>> x = np.array([1.0])  # 1m height
        >>> u = np.array([0.0])   # no inflow
        >>> dx = tank.dynamics(x, u)
        >>> dx[0] < 0  # Height decreases
        True
        >>> tank.energy(x) > 0
        True
    """

    def H(x: npt.NDArray[np.floating]) -> float:
        """Potential energy: (1/2) ρ g A h²."""
        h = x[0]
        return 0.5 * rho * g * A * h**2

    def grad_H(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Gradient: ∇H = ρ g A h."""
        h = x[0]
        return np.array([rho * g * A * h])

    def J(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Interconnection: J = 0 (1D system, no internal coupling)."""
        return np.zeros((1, 1))

    def R(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Dissipation from drain: R = c_d a / (ρ g A²).

        The drain flow is: q_out = c_d a √(2gh) = R ∇H
        where R is chosen so that the PHS structure is exact.
        """
        # For the PHS form, we need: R ∇H = c_d a √(2gh)
        # With ∇H = ρ g A h, we have: R = c_d a √(2gh) / (ρ g A h)
        #                                = c_d a √(2/h) / (ρ g A √h)
        #                                = c_d a √(2) / (ρ g A h)
        # But this makes R state-dependent and singular at h=0.
        # Better: use a regularized/linearized R around nominal h.
        # For simplicity, use constant R assuming nominal operation.
        h = x[0]
        if h < 1e-6:
            # Avoid singularity at h=0
            R_val = 0.0
        else:
            # Linearized dissipation around current height
            R_val = c_d * a * np.sqrt(2.0 / h) / (rho * g * A)
        return np.array([[R_val]])

    def g_mat(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Input map: g = [1/A] (inflow increases volume at rate u/A)."""
        return np.array([[1.0 / A]])

    return PortHamiltonianSystem(
        H=H,
        J=J,
        R=R,
        g=g_mat,
        n_states=1,
        n_inputs=1,
        grad_H=grad_H,
    )


def mass_spring_damper(
    m: float = 1.0,
    k: float = 1.0,
    c: float = 0.1,
) -> PortHamiltonianSystem:
    """
    Mass-spring-damper system (canonical mechanical PHS).

    State: x = [q, p] (position, momentum)
    Input: u = [F] (external force)
    Energy: H(q, p) = (1/2) k q² + (1/2) p²/m (potential + kinetic)

    Dynamics:
        ẋ = [J - R] ∇H + g u

    where:
        J = [[0, 1], [-1, 0]] (canonical symplectic form)
        R = [[0, 0], [0, c]] (damping on momentum)
        g = [[0], [1]] (force acts on momentum)
        ∇H = [k q, p/m]

    Args:
        m: Mass (kg)
        k: Spring constant (N/m)
        c: Damping coefficient (N·s/m)

    Returns:
        PortHamiltonianSystem for mass-spring-damper

    Example:
        >>> sys = mass_spring_damper()
        >>> x = np.array([1.0, 0.0])  # Displaced, at rest
        >>> u = np.array([0.0])
        >>> dx = sys.dynamics(x, u)
        >>> # Position unchanged (p=0), momentum decreases (restoring force)
    """

    def H(x: npt.NDArray[np.floating]) -> float:
        """Total energy: (1/2) k q² + (1/2) p²/m."""
        q, p = x[0], x[1]
        return 0.5 * k * q**2 + 0.5 * p**2 / m

    def grad_H(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Gradient: [k q, p/m]."""
        q, p = x[0], x[1]
        return np.array([k * q, p / m])

    def J(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Canonical symplectic form."""
        return np.array([[0.0, 1.0], [-1.0, 0.0]])

    def R(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Damping on momentum."""
        return np.array([[0.0, 0.0], [0.0, c]])

    def g_mat(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Force acts on momentum."""
        return np.array([[0.0], [1.0]])

    return PortHamiltonianSystem(
        H=H,
        J=J,
        R=R,
        g=g_mat,
        n_states=2,
        n_inputs=1,
        grad_H=grad_H,
    )
