"""
Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests
in the hybrid digital twin framework.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from typing import Dict, Any
import os
import sys

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set environment variables for testing
os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")

# Configure numpy for reproducible tests
np.random.seed(42)

# Suppress TensorFlow warnings during testing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@pytest.fixture
def sample_battery_data():
    """Create realistic sample battery data for testing."""
    np.random.seed(42)
    n_cycles = 100

    # Generate realistic battery degradation data
    cycles = np.arange(1, n_cycles + 1)
    base_temp = 25.0
    temperature = base_temp + np.random.normal(0, 2, n_cycles)
    time = 3600 + np.random.normal(0, 100, n_cycles)
    time = np.clip(time, 1000, 7200)  # Realistic charge times

    # Physics-based capacity degradation
    initial_capacity = 2.0
    k = 0.13
    degradation_factor = k * temperature * cycles / time
    capacity = initial_capacity * np.exp(-degradation_factor * 0.001)  # Scale for realistic degradation

    # Add realistic noise
    capacity += np.random.normal(0, 0.005, n_cycles)
    capacity = np.clip(capacity, 0.5, initial_capacity)  # Ensure physical constraints

    # Additional realistic columns
    voltage = 3.7 - 0.3 * (1 - capacity / initial_capacity) + np.random.normal(0, 0.05, n_cycles)
    current = -2.0 + np.random.normal(0, 0.1, n_cycles)

    return pd.DataFrame({
        'id_cycle': cycles,
        'Temperature_measured': temperature,
        'Time': time,
        'Capacity': capacity,
        'Voltage_measured': voltage,
        'Current_measured': current,
        'Battery': ['B0005'] * n_cycles
    })

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def config():
    """Default test configuration."""
    return {
        "physics_k": 0.13,
        "ml_model": {
            "hidden_layers": [16, 8],  # Smaller for faster testing
            "dropout_rate": 0.1,
            "learning_rate": 0.01,
            "epochs": 5,  # Fewer epochs for testing
            "batch_size": 16,
            "early_stopping_patience": 2
        }
    }

@pytest.fixture(scope="session", autouse=True)
def suppress_warnings():
    """Suppress warnings during testing."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )

@pytest.fixture
def minimal_data():
    """Create minimal valid data for basic testing."""
    return pd.DataFrame({
        'id_cycle': [1, 2, 3, 4, 5],
        'Temperature_measured': [25.0, 25.1, 25.2, 25.3, 25.4],
        'Time': [3600, 3610, 3620, 3630, 3640],
        'Capacity': [2.0, 1.99, 1.98, 1.97, 1.96],
        'Voltage_measured': [3.7, 3.69, 3.68, 3.67, 3.66],
        'Current_measured': [-2.0, -2.0, -2.0, -2.0, -2.0]
    })
