"""
Hybrid Digital Twin Framework for Li-ion Battery Modeling.

A professional-grade implementation combining physics-based modeling with machine learning
for accurate battery capacity prediction and lifecycle management.
"""

__version__ = "1.0.0"
__author__ = "Javier Marin"
__email__ = "javier@example.com"

from hybrid_digital_twin.core.digital_twin import HybridDigitalTwin
from hybrid_digital_twin.models.physics_model import PhysicsBasedModel
from hybrid_digital_twin.models.ml_model import MLCorrectionModel
from hybrid_digital_twin.data.data_loader import BatteryDataLoader
from hybrid_digital_twin.utils.metrics import ModelMetrics

__all__ = [
    "HybridDigitalTwin",
    "PhysicsBasedModel",
    "MLCorrectionModel",
    "BatteryDataLoader",
    "ModelMetrics",
]
