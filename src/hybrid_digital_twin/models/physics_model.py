"""
Physics-based battery degradation model implementation.

This module implements the physics-based component of the hybrid digital twin,
based on the lithium-ion battery degradation model from Xu et al. (2016).
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger

from hybrid_digital_twin.utils.exceptions import ModelError, InvalidParameterError


@dataclass
class PhysicsModelParameters:
    """Parameters for the physics-based model."""

    k: float = 0.13  # Degradation coefficient
    initial_capacity: Optional[float] = None  # Will be inferred from data
    temperature_ref: float = 25.0  # Reference temperature (°C)

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.k <= 0:
            raise InvalidParameterError("Degradation coefficient k must be positive")
        if self.temperature_ref < -50 or self.temperature_ref > 100:
            raise InvalidParameterError("Reference temperature out of realistic range")


class PhysicsBasedModel:
    """
    Physics-based Li-ion battery degradation model.

    This model implements the mathematical framework from Xu et al. (2016):

    Battery lifetime degradation:
    L = 1 - (1 - L') * exp(-f_d)

    Where the degradation factor f_d is defined as:
    f_d = k * T_c * i / t

    And battery capacity evolution follows:
    C(t) = C_0 * exp(-f_d)

    Parameters:
        k: Degradation coefficient (empirically determined as 0.13)
        T_c: Cell temperature (°C)
        i: Cycle number
        t: Charge time per cycle (seconds)
        C_0: Initial battery capacity (Ah)

    References:
        Xu, Bolun, et al. "Modeling of Lithium-Ion Battery Degradation for Cell
        Life Assessment." IEEE Transactions on Smart Grid 7.2 (2016): 826-835.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Initialize the physics-based model.

        Args:
            config: Configuration dictionary containing model parameters
        """
        config = config or {}
        self.params = PhysicsModelParameters(
            k=config.get("physics_k", 0.13),
            temperature_ref=config.get("temperature_ref", 25.0),
        )
        self.is_fitted = False
        self.fit_metrics: Dict = {}

        logger.debug(f"Initialized PhysicsBasedModel with parameters: {self.params}")

    def fit(self, data: pd.DataFrame, target_column: str = "Capacity") -> Dict[str, float]:
        """
        Fit the physics model to training data.

        This method estimates the initial capacity C_0 from the data and validates
        model performance against the physics-based predictions.

        Args:
            data: Training data containing cycle information and capacity measurements
            target_column: Name of the capacity column in the data

        Returns:
            Dictionary containing fit metrics (RMSE, MAE, R²)

        Raises:
            ModelError: If required columns are missing or fitting fails
        """
        try:
            logger.info("Fitting physics-based model")

            # Validate required columns
            required_cols = ['id_cycle', 'Temperature_measured', 'Time', target_column]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ModelError(f"Missing required columns: {missing_cols}")

            # Estimate initial capacity from data
            self.params.initial_capacity = data[target_column].iloc[0]
            logger.debug(f"Estimated initial capacity: {self.params.initial_capacity:.4f} Ah")

            # Generate physics predictions
            physics_pred = self._predict_physics(data)
            actual_capacity = data[target_column].values

            # Calculate fit metrics
            self.fit_metrics = self._calculate_physics_metrics(actual_capacity, physics_pred)

            self.is_fitted = True
            logger.success(f"Physics model fitted successfully. RMSE: {self.fit_metrics['rmse']:.4f}")

            return self.fit_metrics

        except Exception as e:
            logger.error(f"Physics model fitting failed: {str(e)}")
            raise ModelError(f"Physics model fitting failed: {str(e)}") from e

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate physics-based capacity predictions.

        Args:
            data: Input data containing cycle and environmental information

        Returns:
            Array of predicted capacities

        Raises:
            ModelError: If model is not fitted or prediction fails
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before making predictions")

        try:
            return self._predict_physics(data)
        except Exception as e:
            logger.error(f"Physics prediction failed: {str(e)}")
            raise ModelError(f"Physics prediction failed: {str(e)}") from e

    def _predict_physics(self, data: pd.DataFrame) -> np.ndarray:
        """
        Internal method to compute physics-based predictions.

        Implements the degradation model:
        C(t) = C_0 * exp(-f_d)
        where f_d = k * T_c * i / t
        """
        cycles = data['id_cycle'].values
        temperature = data['Temperature_measured'].values
        charge_time = data['Time'].values

        # Handle edge cases
        charge_time = np.where(charge_time <= 0, 1e-6, charge_time)  # Avoid division by zero

        # Calculate degradation factor
        f_d = self.params.k * temperature * cycles / charge_time

        # Apply exponential degradation model
        capacity_predictions = self.params.initial_capacity * np.exp(-f_d)

        return capacity_predictions

    def _calculate_physics_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate physics model performance metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "n_samples": len(y_true),
        }

    def get_degradation_factor(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate the degradation factor f_d for given conditions.

        Args:
            data: Input data with cycle, temperature, and time information

        Returns:
            Array of degradation factors
        """
        cycles = data['id_cycle'].values
        temperature = data['Temperature_measured'].values
        charge_time = data['Time'].values

        charge_time = np.where(charge_time <= 0, 1e-6, charge_time)

        return self.params.k * temperature * cycles / charge_time

    def predict_lifetime(self,
                        max_cycles: int,
                        temperature: float,
                        charge_time: float,
                        capacity_threshold: float = 0.8) -> Dict[str, float]:
        """
        Predict battery lifetime until capacity falls below threshold.

        Args:
            max_cycles: Maximum cycles to simulate
            temperature: Operating temperature (°C)
            charge_time: Charge time per cycle (seconds)
            capacity_threshold: Capacity threshold as fraction of initial capacity

        Returns:
            Dictionary with lifetime predictions
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before lifetime prediction")

        cycles = np.arange(1, max_cycles + 1)

        # Create synthetic data
        data = pd.DataFrame({
            'id_cycle': cycles,
            'Temperature_measured': temperature,
            'Time': charge_time,
        })

        predicted_capacities = self.predict(data)
        normalized_capacities = predicted_capacities / self.params.initial_capacity

        # Find when capacity drops below threshold
        below_threshold = normalized_capacities < capacity_threshold

        if np.any(below_threshold):
            end_of_life_cycle = cycles[np.where(below_threshold)[0][0]]
            end_of_life_capacity = predicted_capacities[below_threshold][0]
        else:
            end_of_life_cycle = max_cycles
            end_of_life_capacity = predicted_capacities[-1]

        return {
            "end_of_life_cycle": int(end_of_life_cycle),
            "end_of_life_capacity": float(end_of_life_capacity),
            "capacity_threshold": capacity_threshold,
            "final_normalized_capacity": float(normalized_capacities[-1]),
            "total_cycles_simulated": max_cycles,
        }

    def sensitivity_analysis(self,
                           data: pd.DataFrame,
                           parameter_ranges: Dict[str, tuple]) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis on model parameters.

        Args:
            data: Reference data for analysis
            parameter_ranges: Dictionary of parameter names and their (min, max) ranges

        Returns:
            Dictionary containing sensitivity results
        """
        results = {}
        base_prediction = self.predict(data)

        for param_name, (min_val, max_val) in parameter_ranges.items():
            if param_name == 'k':
                original_k = self.params.k

                # Test different values of k
                k_values = np.linspace(min_val, max_val, 10)
                predictions = []

                for k_val in k_values:
                    self.params.k = k_val
                    pred = self._predict_physics(data)
                    predictions.append(pred)

                results[param_name] = {
                    'values': k_values,
                    'predictions': np.array(predictions),
                    'rmse_vs_base': [
                        np.sqrt(np.mean((pred - base_prediction) ** 2))
                        for pred in predictions
                    ]
                }

                # Restore original value
                self.params.k = original_k

        return results

    def export_parameters(self) -> Dict:
        """Export model parameters for serialization."""
        return {
            "k": self.params.k,
            "initial_capacity": self.params.initial_capacity,
            "temperature_ref": self.params.temperature_ref,
            "is_fitted": self.is_fitted,
            "fit_metrics": self.fit_metrics,
        }

    def import_parameters(self, params: Dict) -> None:
        """Import model parameters from serialization."""
        self.params.k = params["k"]
        self.params.initial_capacity = params["initial_capacity"]
        self.params.temperature_ref = params["temperature_ref"]
        self.is_fitted = params["is_fitted"]
        self.fit_metrics = params["fit_metrics"]
