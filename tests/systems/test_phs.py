"""Property-based tests for Port-Hamiltonian systems."""

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from otwin.systems.library import dc_motor, mass_spring_damper, water_tank
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
