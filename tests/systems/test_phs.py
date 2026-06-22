"""Property-based tests for Port-Hamiltonian systems."""

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from otwin.systems.library import (
    dc_motor,
    mass_spring_damper,
    pumped_hydro,
    water_tank,
)
from otwin.systems.phs import PortHamiltonianSystem


@given(
    h=st.floats(min_value=0.1, max_value=10.0),
)
@settings(deadline=500)
def test_water_tank_structure(h: float) -> None:
    """Property: Water tank has skew J and PSD R at all valid states."""
    tank = water_tank()
    x = np.array([h])

    structure = tank.check_structure(x)

    # J must be skew-symmetric
    is_skew, violation = structure["J_skew"]
    assert is_skew, f"J not skew-symmetric, violation: {violation}"

    # R must be PSD
    is_psd, min_eig = structure["R_psd"]
    assert is_psd, f"R not PSD, min eigenvalue: {min_eig}"


@given(
    h=st.floats(min_value=0.1, max_value=10.0),
)
@settings(deadline=500)
def test_water_tank_power_balance(h: float) -> None:
    """Property: Power balance identity holds: dH/dt = dissipated + supplied."""
    tank = water_tank()
    x = np.array([h])
    u = np.array([0.0])

    pb = tank.power_balance(x, u)

    # Power balance identity: dH/dt = dissipated + supplied
    expected = pb["dissipated"] + pb["supplied"]
    actual = pb["dH_dt"]

    assert np.isclose(actual, expected, atol=1e-8), (
        f"Power balance violated: dH/dt={actual}, " f"dissipated + supplied={expected}"
    )


@given(
    h=st.floats(min_value=0.1, max_value=10.0),
)
@settings(deadline=500)
def test_water_tank_energy_decay_no_input(h: float) -> None:
    """Property: With u=0, energy is non-increasing (dH/dt ≤ 0)."""
    tank = water_tank()
    x = np.array([h])
    u = np.array([0.0])

    pb = tank.power_balance(x, u)
    dH_dt = pb["dH_dt"]

    # With no input, dissipation dominates
    assert dH_dt <= 1e-10, f"Energy increased with u=0: dH/dt={dH_dt}"


@given(
    h=st.floats(min_value=0.1, max_value=10.0),
)
@settings(deadline=500)
def test_water_tank_dissipation_negative(h: float) -> None:
    """Property: Dissipated power is always non-positive."""
    tank = water_tank()
    x = np.array([h])
    u = np.array([0.0])

    pb = tank.power_balance(x, u)
    dissipated = pb["dissipated"]

    assert dissipated <= 1e-10, f"Dissipation positive: {dissipated}"


@given(
    q=st.floats(min_value=-5.0, max_value=5.0),
    p=st.floats(min_value=-5.0, max_value=5.0),
)
@settings(deadline=500)
def test_mass_spring_damper_structure(q: float, p: float) -> None:
    """Property: Mass-spring-damper has skew J and PSD R."""
    sys = mass_spring_damper()
    x = np.array([q, p])

    structure = sys.check_structure(x)

    is_skew, _ = structure["J_skew"]
    assert is_skew

    is_psd, min_eig = structure["R_psd"]
    assert is_psd, f"R not PSD, min eigenvalue: {min_eig}"


@given(
    q=st.floats(min_value=-5.0, max_value=5.0),
    p=st.floats(min_value=-5.0, max_value=5.0),
)
@settings(deadline=500)
def test_mass_spring_damper_power_balance(q: float, p: float) -> None:
    """Property: Power balance identity holds."""
    sys = mass_spring_damper()
    x = np.array([q, p])
    u = np.array([0.0])

    pb = sys.power_balance(x, u)

    expected = pb["dissipated"] + pb["supplied"]
    actual = pb["dH_dt"]

    assert np.isclose(actual, expected, atol=1e-8)


@given(
    q=st.floats(min_value=-5.0, max_value=5.0),
    p=st.floats(min_value=-5.0, max_value=5.0),
)
@settings(deadline=500)
def test_mass_spring_damper_energy_decay(q: float, p: float) -> None:
    """Property: With u=0, energy is non-increasing."""
    sys = mass_spring_damper()
    x = np.array([q, p])
    u = np.array([0.0])

    pb = sys.power_balance(x, u)

    # With damping, energy should decrease (or stay constant if at rest)
    assert pb["dH_dt"] <= 1e-10


def test_water_tank_trajectory_energy_decay() -> None:
    """Integration test: Energy monotonically decreases over trajectory with u=0."""
    from otwin.integrate import integrate_with_inputs

    tank = water_tank()
    x0 = np.array([2.0])  # Initial height 2m
    t_eval = np.linspace(0, 10, 100)
    u = np.zeros((100, 1))  # No input

    def dynamics(
        t: float, x: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        return tank.dynamics(x, u)

    result = integrate_with_inputs(dynamics, x0, t_eval, u)
    assert result["success"]

    # Compute energy along trajectory
    energies = np.array([tank.energy(x) for x in result["x"]])

    # Energy should monotonically decrease (or stay constant within tolerance)
    energy_diffs = np.diff(energies)
    assert np.all(energy_diffs <= 1e-6), "Energy increased along trajectory"


def test_mass_spring_damper_trajectory_energy_decay() -> None:
    """Integration test: Energy decreases with damping."""
    from otwin.integrate import integrate_with_inputs

    sys = mass_spring_damper(m=1.0, k=1.0, c=0.5)
    x0 = np.array([1.0, 0.0])  # Displaced, at rest
    t_eval = np.linspace(0, 10, 100)
    u = np.zeros((100, 1))

    def dynamics(
        t: float, x: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        return sys.dynamics(x, u)

    result = integrate_with_inputs(dynamics, x0, t_eval, u)
    assert result["success"]

    energies = np.array([sys.energy(x) for x in result["x"]])
    energy_diffs = np.diff(energies)

    # With damping, energy should decrease
    assert np.all(energy_diffs <= 1e-6)

    # Final energy should be less than initial
    assert energies[-1] < energies[0]


@given(
    phi=st.floats(min_value=-5.0, max_value=5.0),
    p=st.floats(min_value=-5.0, max_value=5.0),
)
@settings(deadline=500)
def test_dc_motor_structure(phi: float, p: float) -> None:
    """Property: DC motor has skew J (the gyrator) and PSD R."""
    motor = dc_motor()
    x = np.array([phi, p])

    structure = motor.check_structure(x)

    is_skew, violation = structure["J_skew"]
    assert is_skew, f"J not skew-symmetric, violation: {violation}"

    is_psd, min_eig = structure["R_psd"]
    assert is_psd, f"R not PSD, min eigenvalue: {min_eig}"


@given(
    phi=st.floats(min_value=-5.0, max_value=5.0),
    p=st.floats(min_value=-5.0, max_value=5.0),
)
@settings(deadline=500)
def test_dc_motor_power_balance(phi: float, p: float) -> None:
    """Property: Power balance identity holds for the DC motor."""
    motor = dc_motor()
    x = np.array([phi, p])
    u = np.array([0.0])

    pb = motor.power_balance(x, u)

    expected = pb["dissipated"] + pb["supplied"]
    actual = pb["dH_dt"]

    assert np.isclose(actual, expected, atol=1e-8)


@given(
    phi=st.floats(min_value=-5.0, max_value=5.0),
    p=st.floats(min_value=-5.0, max_value=5.0),
)
@settings(deadline=500)
def test_dc_motor_energy_decay_no_input(phi: float, p: float) -> None:
    """Property: With V=0, the DC motor's stored energy is non-increasing."""
    motor = dc_motor()
    x = np.array([phi, p])
    u = np.array([0.0])

    pb = motor.power_balance(x, u)
    assert pb["dH_dt"] <= 1e-10, f"Energy increased with V=0: dH/dt={pb['dH_dt']}"


def test_dc_motor_steady_state_matches_closed_form() -> None:
    """Integration: numeric steady state matches the analytic omega_ss, I_ss.

    Under a constant voltage V the port-Hamiltonian DC motor converges to
        omega_ss = V*K / (Re*b + K^2),   I_ss = V*b / (Re*b + K^2).
    """
    from otwin.integrate import integrate_with_inputs

    L, inertia, Re, b, K, V = 0.5, 0.01, 1.0, 0.1, 0.5, 10.0
    motor = dc_motor(L=L, inertia=inertia, Re=Re, b=b, K=K)

    x0 = np.array([0.0, 0.0])
    t_eval = np.linspace(0, 6, 600)
    u = np.full((600, 1), V)

    def dynamics(
        t: float, x: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        return motor.dynamics(x, u)

    result = integrate_with_inputs(dynamics, x0, t_eval, u)
    assert result["success"]

    phi_end, p_end = result["x"][-1]
    omega_num = p_end / inertia
    I_num = phi_end / L

    denom = Re * b + K**2
    omega_ss = V * K / denom
    I_ss = V * b / denom

    assert np.isclose(omega_num, omega_ss, rtol=1e-3)
    assert np.isclose(I_num, I_ss, rtol=1e-3)


@given(
    v_u=st.floats(min_value=1.0e5, max_value=2.0e6),
    v_l=st.floats(min_value=1.0e6, max_value=2.0e7),
)
@settings(deadline=500)
def test_pumped_hydro_structure(v_u: float, v_l: float) -> None:
    """Property: pumped hydro has J = 0 (skew) and R PSD (penstock Laplacian)."""
    plant = pumped_hydro()
    x = np.array([v_u, v_l])

    structure = plant.check_structure(x)

    is_skew, violation = structure["J_skew"]
    assert is_skew, f"J not skew-symmetric, violation: {violation}"

    is_psd, min_eig = structure["R_psd"]
    assert is_psd, f"R not PSD, min eigenvalue: {min_eig}"


@given(
    v_u=st.floats(min_value=1.0e5, max_value=2.0e6),
    v_l=st.floats(min_value=1.0e6, max_value=2.0e7),
)
@settings(deadline=500)
def test_pumped_hydro_passivity(v_u: float, v_l: float) -> None:
    """Property: with the pump off the stored energy is non-increasing."""
    plant = pumped_hydro()
    x = np.array([v_u, v_l])
    u = np.array([0.0])

    pb = plant.power_balance(x, u)
    assert pb["dH_dt"] <= 1e-6, f"Energy increased with pump off: dH/dt={pb['dH_dt']}"


def test_pumped_hydro_conservation_when_sealed() -> None:
    """Integration: as the penstock seals (R -> 0), an idle store conserves energy."""
    from otwin.integrate import integrate_with_inputs

    plant = pumped_hydro(R_penstock=1.0e12)  # valve effectively closed
    x0 = np.array([1.0e6, 1.0e7])
    t_eval = np.linspace(0, 6 * 3600.0, 200)
    u = np.zeros((200, 1))  # pump off

    def dynamics(
        t: float, x: npt.NDArray[np.floating], u: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        return plant.dynamics(x, u)

    result = integrate_with_inputs(dynamics, x0, t_eval, u)
    assert result["success"]

    energies = np.array([plant.energy(x) for x in result["x"]])
    # essentially constant: relative drift over 6 hours is negligible
    drift = abs(energies[-1] - energies[0]) / energies[0]
    assert drift < 1e-4, f"Sealed idle store did not conserve energy: drift={drift}"


def test_pumped_hydro_round_trip_efficiency() -> None:
    """Integration: charge then fully discharge; round trip = eta_pump * eta_turbine.

    Conversion losses live at the power port (pump/turbine efficiency); the store
    itself is conservative, so the end-to-end round-trip efficiency equals the
    closed-form product of the two conversion efficiencies.
    """
    from scipy.integrate import solve_ivp

    rho, g, a_u, a_l, z_u = 1000.0, 9.81, 5.0e4, 5.0e6, 300.0
    eta_p, eta_t, p_set = 0.90, 0.90, 100.0e6
    plant = pumped_hydro(A_u=a_u, A_l=a_l, z_u=z_u, g=g, rho=rho)
    v_u0 = 2.0e5
    x0 = np.array([v_u0, 1.2e7])

    def head(x: npt.NDArray[np.floating]) -> float:
        return float((z_u + x[0] / a_u) - (x[1] / a_l))

    def flow(x: npt.NDArray[np.floating], p: float) -> float:
        if p > 0:
            return eta_p * p / (rho * g * head(x))
        if p < 0:
            return p / (eta_t * rho * g * head(x))
        return 0.0

    def make_dyn(p: float):
        def dyn(t: float, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return plant.dynamics(x, np.array([flow(x, p)]))

        return dyn

    hr = 3600.0
    charge = solve_ivp(
        make_dyn(p_set), (0, 8 * hr), x0, max_step=60.0, rtol=1e-8, atol=1e-3
    )
    e_in = p_set * charge.t[-1]

    def back(t: float, x: npt.NDArray[np.floating]) -> float:
        return x[0] - v_u0

    back.terminal = True
    back.direction = -1.0
    discharge = solve_ivp(
        make_dyn(-p_set),
        (0, 12 * hr),
        charge.y[:, -1],
        max_step=60.0,
        rtol=1e-8,
        atol=1e-3,
        events=back,
    )
    e_out = p_set * discharge.t[-1]

    assert np.isclose(e_out / e_in, eta_p * eta_t, rtol=2e-3)
