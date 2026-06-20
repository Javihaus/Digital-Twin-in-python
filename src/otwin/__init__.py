"""
otwin: Composable, physically-consistent (port-Hamiltonian) digital twins
with calibrated uncertainty and rigorous evaluation by default.

Public API (stable):
- DigitalTwin: Main twin interface
- evaluate: Evaluation protocol
- PortHamiltonianSystem: Analytic PHS
- systems.library: Reference systems (water_tank, mass_spring_damper)
"""

__version__ = "2.0.0-alpha"

# Phase 1: PHS core ✅
# Phase 2: Evaluation harness ✅
from otwin.evaluation import EvalReport, evaluate
from otwin.learn import PortHamiltonianNN  # constructs lazily; needs [torch] at use
from otwin.systems import PortHamiltonianSystem, mass_spring_damper, water_tank

# Phase 5: IPHS ✅ (entropy production, second-law enforcement)
from otwin.systems.iphs import IrreversiblePHS

# Phase 3: DigitalTwin ✅
from otwin.twin import DigitalTwin

# Phase 4: Learned PHS + UQ ✅ (core: Ensemble + calibration; PHNN behind [torch])
from otwin.uq import Ensemble

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
