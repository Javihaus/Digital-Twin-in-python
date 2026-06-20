"""Pytest configuration and shared fixtures (v2, core-only).

Deliberately imports nothing beyond the standard library and numpy so the test
session loads under a core install (numpy + scipy only). v1 / pandas-based
fixtures live with the legacy tests under ``legacy_v1/``.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Reproducible default RNG state for any test that does not seed explicitly.
np.random.seed(42)


@pytest.fixture
def temp_dir():
    """A temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session", autouse=True)
def suppress_warnings():
    """Quiet expected warnings during the test session."""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
