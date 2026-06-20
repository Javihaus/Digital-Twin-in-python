"""Time integration utilities for dynamical systems."""

from otwin.integrate.solvers import integrate, integrate_with_inputs
from otwin.integrate.structure_preserving import implicit_midpoint, integrate_phs

__all__ = [
    "integrate",
    "integrate_with_inputs",
    "implicit_midpoint",
    "integrate_phs",
]
