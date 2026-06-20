"""Reproducibility utilities for deterministic seeding."""

from typing import Any

import numpy as np


def set_seed(seed: int) -> dict[str, Any]:
    """
    Set random seeds for reproducibility.

    Seeds numpy's random state. If torch is installed and imported,
    also seeds torch (CPU). Does not guarantee bit-exact reproducibility
    across different hardware or software versions.

    Args:
        seed: Random seed (non-negative integer)

    Returns:
        Dictionary with seeded libraries and any warnings about
        residual nondeterminism

    Raises:
        ValueError: If seed is negative

    Example:
        >>> import otwin.utils.seeding as seeding
        >>> info = seeding.set_seed(42)
        >>> info['numpy']
        True
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    # Always seed numpy
    np.random.seed(seed)
    result: dict[str, Any] = {"numpy": True}

    # Optionally seed torch if available
    try:
        import torch

        torch.manual_seed(seed)
        result["torch"] = True
    except ImportError:
        result["torch"] = False

    # Residual nondeterminism warnings
    result["warnings"] = [
        "CPU operations with different BLAS/LAPACK implementations may vary",
        "Some scipy functions may have nondeterministic behavior",
    ]

    return result
