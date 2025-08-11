"""
Performance benchmarks for the hybrid digital twin framework.

This module contains performance tests to ensure the framework
meets performance requirements.
"""

import pytest
import numpy as np
import time
from unittest.mock import patch

# Handle pytest-benchmark import gracefully
try:
    import pytest_benchmark
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

from hybrid_digital_twin import HybridDigitalTwin
from hybrid_digital_twin.models.physics_model import PhysicsBasedModel
from hybrid_digital_twin.utils.metrics import ModelMetrics


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_physics_model_prediction_speed(self, benchmark, sample_battery_data):
        """Benchmark physics model prediction speed."""
        model = PhysicsBasedModel()
        model.fit(sample_battery_data)

        # Benchmark the prediction
        result = benchmark(model.predict, sample_battery_data)

        # Verify result is reasonable
        assert len(result) == len(sample_battery_data)
        assert np.all(result > 0)

    def test_metrics_calculation_speed(self, benchmark, sample_battery_data):
        """Benchmark metrics calculation speed."""
        metrics_calc = ModelMetrics()
        y_true = sample_battery_data['Capacity'].values
        y_pred = y_true + np.random.normal(0, 0.01, len(y_true))

        # Benchmark metrics calculation
        result = benchmark(metrics_calc.calculate_all_metrics, y_true, y_pred)

        # Verify result structure
        assert isinstance(result, dict)
        assert 'rmse' in result
        assert 'mae' in result
        assert 'r2' in result

    @pytest.mark.slow
    def test_training_speed_small_model(self, benchmark, minimal_data):
        """Benchmark training speed for small model."""
        def train_small_model():
            config = {
                "ml_model": {
                    "hidden_layers": [4],
                    "epochs": 2,
                    "batch_size": 4
                }
            }
            twin = HybridDigitalTwin(config=config)
            return twin.fit(minimal_data, validation_split=0.4)

        # Benchmark the training
        result = benchmark.pedantic(train_small_model, rounds=1, iterations=1)

        # Verify training completed
        assert isinstance(result, dict)

    def test_data_loading_speed(self, benchmark, temp_dir, sample_battery_data):
        """Benchmark data loading speed."""
        from hybrid_digital_twin.data.data_loader import BatteryDataLoader

        # Save test data
        data_file = temp_dir / "benchmark_data.csv"
        sample_battery_data.to_csv(data_file, index=False)

        loader = BatteryDataLoader(data_dir=temp_dir)

        # Benchmark data loading
        result = benchmark(loader.load_csv, data_file.name, validate=True, clean=True)

        # Verify data loaded correctly
        assert len(result) == len(sample_battery_data)
        assert list(result.columns) == list(sample_battery_data.columns)


class TestMemoryUsage:
    """Memory usage tests."""

    def test_model_memory_footprint(self, sample_battery_data):
        """Test that model doesn't use excessive memory."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and train model
        twin = HybridDigitalTwin(config={
            "ml_model": {"hidden_layers": [32, 16], "epochs": 5}
        })

        twin.fit(sample_battery_data, validation_split=0.2)

        # Make predictions
        predictions = twin.predict(sample_battery_data)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use more than 500MB additional memory for test data
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f} MB"

        # Verify functionality
        assert len(predictions) == len(sample_battery_data)


class TestScalability:
    """Scalability tests."""

    @pytest.mark.parametrize("data_size", [50, 100])
    def test_prediction_scales_linearly(self, data_size):
        """Test that prediction time scales reasonably with data size."""
        # Create synthetic data of different sizes
        cycles = np.arange(1, data_size + 1)
        data = {
            'id_cycle': cycles,
            'Temperature_measured': np.full(data_size, 25.0),
            'Time': np.full(data_size, 3600.0),
            'Capacity': 2.0 * np.exp(-0.001 * cycles),
            'Voltage_measured': np.full(data_size, 3.7),
            'Current_measured': np.full(data_size, -2.0)
        }

        import pandas as pd
        test_data = pd.DataFrame(data)

        # Train a simple model
        model = PhysicsBasedModel()
        model.fit(test_data)

        # Time the prediction
        start_time = time.time()
        predictions = model.predict(test_data)
        prediction_time = time.time() - start_time

        # Basic scalability check
        time_per_sample = prediction_time / data_size
        assert time_per_sample < 0.01, f"Prediction too slow: {time_per_sample:.4f}s per sample"
        assert len(predictions) == data_size


# Mock benchmark if pytest-benchmark is not available
def mock_benchmark_fixture():
    """Mock benchmark fixture for when pytest-benchmark is not available."""
    class MockBenchmark:
        def __call__(self, func, *args, **kwargs):
            return func(*args, **kwargs)

        def pedantic(self, func, rounds=1, iterations=1):
            return func()

    return MockBenchmark()


# Provide benchmark fixture if not available
@pytest.fixture
def benchmark():
    """Benchmark fixture that works with or without pytest-benchmark."""
    try:
        # This will be overridden by pytest-benchmark if available
        return mock_benchmark_fixture()
    except ImportError:
        return mock_benchmark_fixture()
