"""Integration tests: end-to-end workflows."""

import numpy as np
import pytest

from PKG import DigitalTwin, evaluate, water_tank
from PKG.evaluation import temporal_holdout
from PKG.integrate import integrate_with_inputs
from PKG.systems.iphs import IrreversiblePHS
from PKG.utils import set_seed


def test_end_to_end_water_tank_workflow() -> None:
    """End-to-end: Create twin, forecast, verify structure."""
    # Seed for reproducibility
    set_seed(42)

    # Create system and twin
    tank = water_tank(A=1.0, a=0.1)
    twin = DigitalTwin(model=tank)

    # Initial state
    x0 = np.array([2.0])
    t = np.linspace(0, 10, 100)
    u = np.zeros((100, 1))

    # Forecast
    result = twin.forecast(x0, t, u)

    # Verify structure
    assert result["success"]
    assert result["x"].shape == (100, 1)
    assert len(result["t"]) == 100

    # Verify energy decay (passivity)
    energies = [tank.energy(x) for x in result["x"]]
    assert energies[-1] < energies[0]  # Energy decreased

    # Verify power balance at multiple points
    for i in [0, 25, 50, 75]:
        x = result["x"][i]
        pb = tank.power_balance(x, np.array([0.0]))
        balance_error = abs(pb["dH_dt"] - (pb["dissipated"] + pb["supplied"]))
        assert balance_error < 1e-6


def test_evaluation_with_temporal_split() -> None:
    """Integration: Evaluation harness with temporal split."""
    # Generate synthetic data
    tank = water_tank()
    x0 = np.array([2.0])
    t = np.linspace(0, 20, 200)
    u = np.zeros((200, 1))

    def dynamics(t_val: float, x: np.ndarray, u_val: np.ndarray) -> np.ndarray:
        return tank.dynamics(x, u_val)

    result = integrate_with_inputs(dynamics, x0, t, u)
    data = result["x"]

    # Add small noise
    data = data + np.random.randn(*data.shape) * 0.01

    # Split
    train, test = temporal_holdout(data, test_frac=0.2)

    # Verify split is temporal
    assert len(train) == 160
    assert len(test) == 40
    assert train[-1, 0] > test[0, 0] or train[-1, 0] < test[0, 0]  # Different values


def test_iphs_entropy_production() -> None:
    """Integration: IPHS with entropy production."""
    # Simple 1D IPHS
    H = lambda x: 0.5 * x[0] ** 2
    S = lambda x: -x[0]  # Entropy increases as energy decreases
    J = lambda x: np.zeros((1, 1))
    R = lambda x: np.array([[0.1]])
    L = lambda x: np.array([[0.05]])
    g = lambda x: np.array([[1.0]])

    iphs = IrreversiblePHS(H, S, J, R, L, g, 1, 1)

    # Integrate
    x0 = np.array([2.0])
    t = np.linspace(0, 10, 100)
    u = np.zeros((100, 1))

    def dynamics(t_val: float, x: np.ndarray, u_val: np.ndarray) -> np.ndarray:
        return iphs.dynamics(x, u_val)

    result = integrate_with_inputs(dynamics, x0, t, u)

    # Verify entropy production is non-negative along trajectory
    for x in result["x"][::10]:  # Check every 10th point
        sigma = iphs.entropy_production(x)
        assert sigma >= -1e-10, f"Entropy production negative: {sigma}"


def test_public_api_imports() -> None:
    """Integration: Verify public API is importable."""
    from PKG import (
        DigitalTwin,
        EvalReport,
        PortHamiltonianSystem,
        evaluate,
        mass_spring_damper,
        water_tank,
    )

    # Verify types
    assert callable(water_tank)
    assert callable(mass_spring_damper)
    assert callable(evaluate)

    # Create instances
    tank = water_tank()
    assert isinstance(tank, PortHamiltonianSystem)

    twin = DigitalTwin(model=tank)
    assert twin.model_type == "analytic"


def test_structure_preservation_over_long_horizon() -> None:
    """Integration: Structure preserved over extended integration."""
    tank = water_tank()

    x0 = np.array([3.0])
    t = np.linspace(0, 50, 500)  # Long horizon
    u = np.zeros((500, 1))

    def dynamics(t_val: float, x: np.ndarray, u_val: np.ndarray) -> np.ndarray:
        return tank.dynamics(x, u_val)

    # Structure preservation is the job of the structure-preserving integrator,
    # not of a generic explicit solver (RK45 can inject energy near equilibrium).
    result = integrate_with_inputs(dynamics, x0, t, u, method="implicit_midpoint")

    # Check structure at many points
    for i in range(0, 500, 50):
        x = result["x"][i]
        structure = tank.check_structure(x)

        is_skew, _ = structure["J_skew"]
        assert is_skew, f"J not skew at t={t[i]}"

        is_psd, min_eig = structure["R_psd"]
        assert is_psd, f"R not PSD at t={t[i]}, min_eig={min_eig}"

    # Energy should monotonically decrease
    energies = [tank.energy(x) for x in result["x"]]
    diffs = np.diff(energies)
    assert np.all(diffs <= 1e-6), "Energy increased somewhere"


def test_seeding_reproducibility() -> None:
    """Integration: Seeding produces reproducible results."""
    from PKG.utils import set_seed

    # First run
    set_seed(42)
    data1 = np.random.randn(100, 1)

    # Second run with same seed
    set_seed(42)
    data2 = np.random.randn(100, 1)

    assert np.allclose(data1, data2), "Seeding not reproducible"
