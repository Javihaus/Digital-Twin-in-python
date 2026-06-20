"""Smoke tests for basic package functionality."""

import otwin
import otwin.utils


def test_package_import() -> None:
    """Test that the package can be imported."""
    assert otwin.__version__ == "2.0.0-alpha"


def test_set_seed() -> None:
    """Test that seeding utility works."""
    from otwin.utils import set_seed

    result = set_seed(42)
    assert result["numpy"] is True
    assert "warnings" in result


def test_set_seed_negative_raises() -> None:
    """Test that negative seeds raise ValueError."""
    from otwin.utils import set_seed
    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)
