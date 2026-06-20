"""
Main Hybrid Digital Twin implementation.

This module contains the core HybridDigitalTwin class that orchestrates
the physics-based model and machine learning correction components.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from hybrid_digital_twin.data.data_loader import BatteryDataLoader
from hybrid_digital_twin.models.ml_model import MLCorrectionModel
from hybrid_digital_twin.models.physics_model import PhysicsBasedModel
from hybrid_digital_twin.utils.exceptions import (
    DigitalTwinError,
    InvalidDataError,
    ModelNotTrainedError,
)
from hybrid_digital_twin.utils.metrics import ModelMetrics
from hybrid_digital_twin.utils.validators import validate_input_data


@dataclass
class PredictionResult:
    """Container for prediction results."""

    physics_prediction: np.ndarray
    ml_correction: np.ndarray
    hybrid_prediction: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class HybridDigitalTwin:
    """
    Hybrid Digital Twin for Li-ion Battery Capacity Prediction.

    This class implements a hybrid approach combining physics-based modeling
    with machine learning corrections to achieve accurate battery capacity
    predictions throughout the battery lifecycle.

    The hybrid approach follows the mathematical framework:

    1. Physics Model: C_physics = C_0 * exp(-f_d)
       where f_d = k * T_c * i / t

    2. ML Correction: ΔC = f_ML(C_physics, features)

    3. Hybrid Prediction: C_hybrid = C_physics + ΔC

    Attributes:
        physics_model: Physics-based degradation model
        ml_model: Machine learning correction model
        is_trained: Flag indicating if models are trained
        metrics: Model performance metrics
    """

    def __init__(
        self,
        physics_model: Optional[PhysicsBasedModel] = None,
        ml_model: Optional[MLCorrectionModel] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the Hybrid Digital Twin.

        Args:
            physics_model: Optional pre-configured physics model
            ml_model: Optional pre-configured ML model
            config: Configuration dictionary
        """
        self.config = config or {}
        self.physics_model = physics_model or PhysicsBasedModel(config=self.config)
        self.ml_model = ml_model or MLCorrectionModel(config=self.config)
        self.is_trained = False
        self.metrics = ModelMetrics()
        self.training_history: Dict = {}

        logger.info("Initialized Hybrid Digital Twin")

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "Capacity",
        validation_split: float = 0.2,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Train the hybrid digital twin on battery data.

        Args:
            data: Battery discharge/cycle data
            target_column: Name of the capacity column
            validation_split: Fraction of data for validation
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training metrics

        Raises:
            InvalidDataError: If input data is invalid
            DigitalTwinError: If training fails
        """
        try:
            logger.info("Starting hybrid digital twin training")

            # Validate input data
            validate_input_data(data, required_columns=[target_column])

            # Split data
            split_idx = int(len(data) * (1 - validation_split))
            train_data = data.iloc[:split_idx]
            val_data = data.iloc[split_idx:]

            # Step 1: Train physics model
            logger.info("Training physics-based model")
            physics_metrics = self.physics_model.fit(train_data, target_column)

            # Step 2: Generate physics predictions
            train_physics_pred = self.physics_model.predict(train_data)
            val_physics_pred = self.physics_model.predict(val_data)

            # Step 3: Calculate residuals for ML model
            train_residuals = train_data[target_column].values - train_physics_pred
            val_residuals = val_data[target_column].values - val_physics_pred

            # Step 4: Train ML correction model
            logger.info("Training ML correction model")
            ml_features_train = self._extract_ml_features(
                train_data, train_physics_pred
            )
            ml_features_val = self._extract_ml_features(val_data, val_physics_pred)

            ml_metrics = self.ml_model.fit(
                ml_features_train,
                train_residuals,
                validation_data=(ml_features_val, val_residuals),
                **kwargs,
            )

            # Step 5: Evaluate hybrid model
            hybrid_pred_train = train_physics_pred + self.ml_model.predict(
                ml_features_train
            )
            hybrid_pred_val = val_physics_pred + self.ml_model.predict(ml_features_val)

            # Calculate metrics
            training_metrics = self._calculate_metrics(
                train_data[target_column].values,
                val_data[target_column].values,
                hybrid_pred_train,
                hybrid_pred_val,
            )

            # Store training history
            self.training_history = {
                "physics_metrics": physics_metrics,
                "ml_metrics": ml_metrics,
                "hybrid_metrics": training_metrics,
                "training_size": len(train_data),
                "validation_size": len(val_data),
            }

            self.is_trained = True
            logger.success("Hybrid digital twin training completed")

            return training_metrics

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise DigitalTwinError(f"Training failed: {str(e)}") from e

    def predict(
        self,
        data: pd.DataFrame,
        return_uncertainty: bool = False,
        return_components: bool = False,
    ) -> Union[np.ndarray, PredictionResult]:
        """
        Make predictions using the hybrid digital twin.

        Args:
            data: Input data for prediction
            return_uncertainty: Whether to compute prediction uncertainty
            return_components: Whether to return individual model components

        Returns:
            Predictions array or detailed PredictionResult object

        Raises:
            ModelNotTrainedError: If models are not trained
            InvalidDataError: If input data is invalid
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Models must be trained before prediction")

        try:
            validate_input_data(data)

            # Physics model prediction
            physics_pred = self.physics_model.predict(data)

            # ML correction
            ml_features = self._extract_ml_features(data, physics_pred)
            ml_correction = self.ml_model.predict(ml_features)

            # Hybrid prediction
            hybrid_pred = physics_pred + ml_correction

            # Uncertainty estimation (if requested)
            uncertainty = None
            if return_uncertainty:
                uncertainty = self._estimate_uncertainty(ml_features)

            if return_components:
                return PredictionResult(
                    physics_prediction=physics_pred,
                    ml_correction=ml_correction,
                    hybrid_prediction=hybrid_pred,
                    uncertainty=uncertainty,
                    metadata={
                        "n_samples": len(data),
                        "feature_dimensions": (
                            ml_features.shape[1] if ml_features.ndim > 1 else 1
                        ),
                    },
                )

            return hybrid_pred

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise DigitalTwinError(f"Prediction failed: {str(e)}") from e

    def predict_future(
        self,
        cycles: np.ndarray,
        temperature: float,
        charge_time: float,
        initial_capacity: float,
        return_uncertainty: bool = False,
    ) -> Union[np.ndarray, PredictionResult]:
        """
        Predict future battery capacity for given cycles.

        Args:
            cycles: Array of future cycle numbers
            temperature: Operating temperature (°C)
            charge_time: Charge time per cycle (seconds)
            initial_capacity: Initial battery capacity (Ah)
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Future capacity predictions
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Models must be trained before prediction")

        # Create synthetic data for future cycles
        future_data = pd.DataFrame(
            {
                "id_cycle": cycles,
                "Temperature_measured": temperature,
                "Time": charge_time,
                "Capacity": initial_capacity,  # Will be overridden by physics model
            }
        )

        return self.predict(
            future_data, return_uncertainty=return_uncertainty, return_components=True
        )

    def evaluate(
        self, test_data: pd.DataFrame, target_column: str = "Capacity"
    ) -> Dict[str, float]:
        """
        Evaluate the hybrid model on test data.

        Args:
            test_data: Test dataset
            target_column: Target column name

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Models must be trained before evaluation")

        predictions = self.predict(test_data)
        actual = test_data[target_column].values

        return self.metrics.calculate_all_metrics(actual, predictions)

    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ModelNotTrainedError("Cannot save untrained model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "physics_model": self.physics_model,
            "ml_model": self.ml_model,
            "training_history": self.training_history,
            "config": self.config,
            "version": "1.0.0",
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "HybridDigitalTwin":
        """Load a trained model from disk."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        instance = cls(
            physics_model=model_data["physics_model"],
            ml_model=model_data["ml_model"],
            config=model_data.get("config", {}),
        )

        instance.training_history = model_data.get("training_history", {})
        instance.is_trained = True

        logger.info(f"Model loaded from {filepath}")
        return instance

    def _extract_ml_features(
        self, data: pd.DataFrame, physics_pred: np.ndarray
    ) -> np.ndarray:
        """Extract features for ML model."""
        features = []

        # Physics prediction as feature
        features.append(physics_pred.reshape(-1, 1))

        # Additional engineered features
        if "Temperature_measured" in data.columns:
            features.append(data["Temperature_measured"].values.reshape(-1, 1))

        if "id_cycle" in data.columns:
            features.append(data["id_cycle"].values.reshape(-1, 1))

        if "Time" in data.columns:
            features.append(data["Time"].values.reshape(-1, 1))

        return np.hstack(features)

    def _estimate_uncertainty(self, features: np.ndarray) -> np.ndarray:
        """Estimate prediction uncertainty using ensemble or dropout."""
        # Simplified uncertainty estimation
        # In production, this could use ensemble methods or Monte Carlo dropout
        return np.ones(len(features)) * 0.01  # Placeholder

    def _calculate_metrics(
        self,
        y_train_true: np.ndarray,
        y_val_true: np.ndarray,
        y_train_pred: np.ndarray,
        y_val_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        metrics = {}

        # Training metrics
        train_metrics = self.metrics.calculate_all_metrics(y_train_true, y_train_pred)
        for key, value in train_metrics.items():
            metrics[f"train_{key}"] = value

        # Validation metrics
        val_metrics = self.metrics.calculate_all_metrics(y_val_true, y_val_pred)
        for key, value in val_metrics.items():
            metrics[f"val_{key}"] = value

        return metrics
