"""
PKG: Composable, physically-consistent (port-Hamiltonian) digital twins
with calibrated uncertainty and honest evaluation by default.

Public API (stable):
- DigitalTwin: Main twin interface
- evaluate: Honest evaluation protocol
- PortHamiltonianSystem: Analytic PHS
- systems.library: Reference systems (water_tank, mass_spring_damper)
"""

__version__ = "2.0.0-alpha"

# Phase 1: PHS core ✅
from PKG.systems import PortHamiltonianSystem, water_tank, mass_spring_damper

# Phase 2: Evaluation harness ✅
from PKG.evaluation import evaluate, EvalReport

# Phase 3: DigitalTwin ✅
from PKG.twin import DigitalTwin

# Phase 4: Learned PHS + UQ ✅ (core: Ensemble + calibration; PHNN behind [torch])
from PKG.uq import Ensemble
from PKG.learn import PortHamiltonianNN  # constructs lazily; needs [torch] at use

# Phase 5: IPHS ✅ (entropy production, second-law enforcement)
from PKG.systems.iphs import IrreversiblePHS

# Phase 6: Composition + GP-PHS (in-progress: GP-PHS done behind [gp];
#          twin/compose.py still a stub)

__all__ = [
    "__version__",
    "DigitalTwin",
    "evaluate",
    "EvalReport",
    "PortHamiltonianSystem",
    "water_tank",
    "mass_spring_damper",
    "Ensemble",
    "PortHamiltonianNN",
    "IrreversiblePHS",
]
