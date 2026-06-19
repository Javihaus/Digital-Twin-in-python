"""Generate reproducible benchmarks for PKG v2."""

import json
from pathlib import Path

import numpy as np

from PKG import water_tank
from PKG.evaluation import evaluate, EvalReport
from PKG.integrate import integrate_with_inputs
from PKG.utils import set_seed

# Ensure results directory exists
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def generate_synthetic_data(seed: int = 42) -> np.ndarray:
    """Generate synthetic water tank data."""
    set_seed(seed)

    tank = water_tank(A=1.0, a=0.1, g=9.81, c_d=0.6)
    x0 = np.array([2.0])
    t = np.linspace(0, 50, 500)
    u = np.zeros((500, 1))

    def dynamics(t_val: float, x: np.ndarray, u_val: np.ndarray) -> np.ndarray:
        return tank.dynamics(x, u_val)

    result = integrate_with_inputs(dynamics, x0, t, u)

    # Add realistic noise
    noise = np.random.randn(*result["x"].shape) * 0.02
    return result["x"] + noise


class WaterTankModel:
    """Simple water tank predictor for benchmarking."""

    def __init__(self) -> None:
        self.tank = water_tank(A=1.0, a=0.1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """One-step predictions."""
        predictions = []
        dt = 0.1  # Assume 0.1s timestep

        for x in X:
            u = np.array([0.0])
            dx = self.tank.dynamics(x, u)
            x_next = x + dx * dt
            predictions.append(x_next)

        return np.array(predictions)


def benchmark_water_tank() -> None:
    """Benchmark water tank PHS model."""
    print("=" * 60)
    print("Benchmark: Water Tank (Analytic PHS)")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(seed=42)

    # Create model
    model = WaterTankModel()

    # Evaluate with rolling-origin protocol
    report = evaluate(
        model,
        data,
        protocol="rolling_origin",
        n_folds=5,
        test_frac=0.2,
        horizon=20,
        seed=42,
    )

    # Print report
    print(report)

    # Save to JSON
    output_file = RESULTS_DIR / "water_tank.json"
    report.to_json(str(output_file))
    print(f"\n✅ Saved to {output_file}")

    # Also save metadata
    metadata = {
        "description": "Water tank PHS with rolling-origin evaluation",
        "system": "Water tank (analytic port-Hamiltonian)",
        "protocol": "rolling_origin",
        "n_folds": 5,
        "data_points": len(data),
        "seed": 42,
        "version": "2.0.0-alpha",
    }

    metadata_file = RESULTS_DIR / "water_tank_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def benchmark_simple_forecasting() -> None:
    """Simple forecasting benchmark with synthetic data."""
    print("\n" + "=" * 60)
    print("Benchmark: Simple Forecasting (Temporal Holdout)")
    print("=" * 60)

    # Generate simple time series
    set_seed(42)
    t = np.linspace(0, 10, 200)
    signal = np.sin(2 * np.pi * 0.5 * t) + np.random.randn(200) * 0.1
    data = signal.reshape(-1, 1)

    # Simple model: persistence baseline
    class PersistenceModel:
        def predict(self, X: np.ndarray) -> np.ndarray:
            # Repeat first value
            return np.full_like(X, X[0])

    model = PersistenceModel()

    # Evaluate
    report = evaluate(
        model,
        data,
        protocol="temporal_holdout",
        test_frac=0.2,
        seed=42,
    )

    print(report)

    # Save
    output_file = RESULTS_DIR / "simple_forecasting.json"
    report.to_json(str(output_file))
    print(f"\n✅ Saved to {output_file}")


def generate_benchmark_summary() -> None:
    """Generate summary of all benchmarks."""
    print("\n" + "=" * 60)
    print("Generating Benchmark Summary")
    print("=" * 60)

    summary = {
        "version": "2.0.0-alpha",
        "benchmarks": [],
    }

    # Load all benchmark results
    for result_file in RESULTS_DIR.glob("*.json"):
        if "metadata" in result_file.name or "summary" in result_file.name:
            continue

        with open(result_file) as f:
            result = json.load(f)

        benchmark_info = {
            "name": result_file.stem,
            "protocol": result.get("split_protocol"),
            "n_folds": result.get("n_folds"),
            "skill_score_rmse": result.get("skill_score_rmse"),
            "baseline": result.get("baseline_name"),
            "model_rmse": result.get("point_metrics", {}).get("rmse"),
            "baseline_rmse": result.get("baseline_metrics", {}).get("rmse"),
        }
        summary["benchmarks"].append(benchmark_info)

    # Save summary
    summary_file = RESULTS_DIR / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved to {summary_file}")

    # Print summary table
    print("\nBenchmark Summary:")
    print("-" * 80)
    print(f"{'Benchmark':<30} {'Protocol':<20} {'Skill Score':<15} {'Baseline':<15}")
    print("-" * 80)

    for bench in summary["benchmarks"]:
        name = bench["name"]
        protocol = bench["protocol"]
        skill = bench["skill_score_rmse"]
        baseline = bench["baseline"]

        skill_str = f"{skill:+.3f}" if skill is not None else "N/A"
        print(f"{name:<30} {protocol:<20} {skill_str:<15} {baseline:<15}")

    print("-" * 80)


if __name__ == "__main__":
    print("PKG v2 - Reproducible Benchmarks")
    print("=" * 60)
    print()

    # Run benchmarks
    benchmark_water_tank()
    benchmark_simple_forecasting()

    # Generate summary
    generate_benchmark_summary()

    print("\n" + "=" * 60)
    print("✅ All benchmarks complete!")
    print("=" * 60)
    print(f"\nResults saved in: {RESULTS_DIR}")
    print("\nTo use in documentation:")
    print("  - Numbers are traceable to seed and data")
    print("  - JSON files contain full evaluation details")
    print("  - Never hand-type these numbers in README")
