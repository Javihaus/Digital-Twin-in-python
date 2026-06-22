"""Library of reference port-Hamiltonian systems."""

import numpy as np
import numpy.typing as npt

from otwin.systems.phs import PortHamiltonianSystem


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
        return float(0.5 * rho * g * A * h**2)

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
        return float(0.5 * k * q**2 + 0.5 * p**2 / m)

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


def dc_motor(
    L: float = 0.5,
    inertia: float = 0.01,
    Re: float = 1.0,
    b: float = 0.1,
    K: float = 0.5,
) -> PortHamiltonianSystem:
    """
    DC motor as a multi-domain (electrical + mechanical) port-Hamiltonian system.

    Reference: van der Schaft & Jeltsema (2014), "Port-Hamiltonian Systems
    Theory: An Introductory Overview", Example 2.5, Eq. (2.30).

    State: x = [phi, p]
        phi - inductor flux-linkage (Wb), electrical energy store
        p   - rotor angular momentum (kg·m²/s), mechanical energy store
    Input: u = [V] (applied voltage)
    Output: y = I (armature current)
    Energy: H(phi, p) = phi² / (2 L) + p² / (2 inertia)

    Dynamics (Eq. 2.30):
        [phi_dot]   ([ 0  -K ]   [Re  0 ]) [phi/L]   [1]
        [ p_dot ] = ([ K   0 ] - [ 0  b ]) [ p/J ] + [0] V
        y = [1 0] grad_H = phi / L

    Structure:
        J = [[0, -K], [K, 0]]   skew-symmetric (the gyrator couples the
                                electrical and mechanical domains)
        R = diag(Re, b)         PSD: armature resistance and viscous friction
        g = [[1], [0]]          voltage drives the electrical state

    With V = 0 the stored energy is non-increasing (passivity by construction).

    Args:
        L: Armature inductance (H)
        inertia: Rotor moment of inertia (kg·m²)
        Re: Armature resistance (ohm)
        b: Viscous friction coefficient (N·m·s)
        K: Motor/gyrator constant (N·m/A = V·s)

    Returns:
        PortHamiltonianSystem for the DC motor.

    Example:
        >>> motor = dc_motor()
        >>> x = np.array([1.0, 0.0])   # some flux, rotor at rest
        >>> u = np.array([0.0])        # no applied voltage
        >>> dx = motor.dynamics(x, u)
        >>> motor.energy(x) > 0
        True
    """

    def H(x: npt.NDArray[np.floating]) -> float:
        """Total energy: magnetic phi²/(2L) + kinetic p²/(2 inertia)."""
        phi, p = x[0], x[1]
        return float(0.5 * phi**2 / L + 0.5 * p**2 / inertia)

    def grad_H(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Gradient: [phi/L, p/inertia] = [current I, angular velocity omega]."""
        phi, p = x[0], x[1]
        return np.array([phi / L, p / inertia])

    def J(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Gyrator coupling (skew-symmetric)."""
        return np.array([[0.0, -K], [K, 0.0]])

    def R(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Dissipation: armature resistance Re and viscous friction b."""
        return np.array([[Re, 0.0], [0.0, b]])

    def g_mat(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Voltage acts on the electrical (flux) state."""
        return np.array([[1.0], [0.0]])

    return PortHamiltonianSystem(
        H=H,
        J=J,
        R=R,
        g=g_mat,
        n_states=2,
        n_inputs=1,
        grad_H=grad_H,
    )


def pumped_hydro(
    A_u: float = 5.0e4,
    A_l: float = 5.0e6,
    z_u: float = 300.0,
    R_penstock: float = 5.0e8,
    g: float = 9.81,
    rho: float = 1000.0,
) -> PortHamiltonianSystem:
    """Pumped-hydro energy storage as a two-reservoir port-Hamiltonian system.

    Pumped hydro is the dominant grid-scale storage technology (~95% of the
    world's installed long-duration storage). Two reservoirs exchange water
    through a reversible pump-turbine; energy is stored as gravitational
    potential energy. Unlike the water tank — whose open drain *dissipates*
    energy — the connection here is a *controlled, reversible* power port, so the
    store is conservative: this is the white-box guarantee a degradation
    (grey-box) model cannot offer.

    State: x = [V_u, V_l] (reservoir volumes, m^3): upper and lower.
    Input: u = [q] (commanded pump-turbine flow, m^3/s); q > 0 pumps water up
        (charging), q < 0 generates (discharging).
    Energy (gravitational potential energy about the lower datum):
        H = rho g [ z_u V_u + V_u^2 / (2 A_u) + V_l^2 / (2 A_l) ]
    so the gradient is the pressure head at each free surface:
        grad_H = rho g [ z_u + V_u/A_u , V_l/A_l ].

    Structure:
        J = 0                         no lossless internal circulation
        R = (1/R_penstock) [[1, -1], [-1, 1]]
                                      penstock conductance (graph-Laplacian, PSD):
                                      with the valve open and the pump off, water
                                      runs downhill and the head difference is
                                      dissipated -- passivity. Default R_penstock is
                                      large, i.e. a near-lossless store.
        g = [[1], [-1]]               the pump-turbine moves water between reservoirs.

    With the pump off the stored energy can only decrease (dH/dt <= 0); as
    R_penstock -> inf it is exactly conserved (a perfect store). Round-trip losses
    in a real plant come from the pump/turbine conversion efficiency, applied at the
    power port (see the worked example), not from the stored energy decaying.

    Args:
        A_u: Upper reservoir surface area (m^2).
        A_l: Lower reservoir surface area (m^2); large => near-constant tailwater.
        z_u: Elevation of the upper reservoir base above the lower datum (m).
        R_penstock: Hydraulic resistance of the (open) penstock (Pa*s/m^3).
        g: Gravitational acceleration (m/s^2).
        rho: Water density (kg/m^3).

    Returns:
        PortHamiltonianSystem for the pumped-hydro store.

    Example:
        >>> plant = pumped_hydro()
        >>> x = np.array([1.0e6, 1.0e7])   # some water up top
        >>> plant.energy(x) > 0
        True
        >>> float(plant.power_balance(x, np.array([0.0]))["dH_dt"]) <= 1e-6
        True
    """

    def H(x: npt.NDArray[np.floating]) -> float:
        """Gravitational potential energy of both reservoirs (J)."""
        v_u, v_l = x[0], x[1]
        return float(
            rho * g * (z_u * v_u + v_u**2 / (2.0 * A_u) + v_l**2 / (2.0 * A_l))
        )

    def grad_H(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Pressure head at each free surface: rho g x surface elevation."""
        v_u, v_l = x[0], x[1]
        return np.array([rho * g * (z_u + v_u / A_u), rho * g * (v_l / A_l)])

    def J(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """No lossless circulation."""
        return np.zeros((2, 2))

    def R(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Penstock conductance (graph-Laplacian, positive semidefinite)."""
        c = 1.0 / R_penstock
        return np.array([[c, -c], [-c, c]])

    def g_mat(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Pump-turbine: +flow into the upper reservoir, -flow out of the lower."""
        return np.array([[1.0], [-1.0]])

    return PortHamiltonianSystem(
        H=H,
        J=J,
        R=R,
        g=g_mat,
        n_states=2,
        n_inputs=1,
        grad_H=grad_H,
    )
