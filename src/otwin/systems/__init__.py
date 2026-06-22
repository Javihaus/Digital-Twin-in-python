"""Port-Hamiltonian systems and reference library."""

from otwin.systems.iphs import IrreversiblePHS
from otwin.systems.library import (
    dc_motor,
    mass_spring_damper,
    pumped_hydro,
    water_tank,
)
from otwin.systems.phs import PortHamiltonianSystem

__all__ = [
    "PortHamiltonianSystem",
    "IrreversiblePHS",
    "water_tank",
    "mass_spring_damper",
    "dc_motor",
    "pumped_hydro",
]
