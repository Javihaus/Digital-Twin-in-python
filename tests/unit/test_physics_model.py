"""
Unit tests for the PhysicsBasedModel class.

This module contains comprehensive unit tests for the physics-based
battery degradation model component.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from hybrid_digital_twin.models.physics_model import (
    PhysicsBasedModel,
    PhysicsModelParameters
)
from hybrid_digital_twin.utils.exceptions import ModelError, InvalidParameterError


class TestPhysicsModelParameters:
    """Test suite for PhysicsModelParameters dataclass."""
    
    def test_default_parameters(self):
        """Test default parameter initialization."""
        params = PhysicsModelParameters()
        
        assert params.k == 0.13
        assert params.initial_capacity is None
        assert params.temperature_ref == 25.0
    
    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        params = PhysicsModelParameters(
            k=0.15,
            initial_capacity=2.5,
            temperature_ref=30.0
        )
        
        assert params.k == 0.15
        assert params.initial_capacity == 2.5
        assert params.temperature_ref == 30.0
    
    def test_invalid_k_parameter(self):
        """Test validation of k parameter."""
        with pytest.raises(InvalidParameterError, match="Degradation coefficient k must be positive"):
            PhysicsModelParameters(k=-0.1)
        
        with pytest.raises(InvalidParameterError, match="Degradation coefficient k must be positive"):
            PhysicsModelParameters(k=0.0)
    
    def test_invalid_temperature_parameter(self):
        """Test validation of temperature parameter."""
        with pytest.raises(InvalidParameterError, match="Reference temperature out of realistic range"):
            PhysicsModelParameters(temperature_ref=-100.0)
        
        with pytest.raises(InvalidParameterError, match="Reference temperature out of realistic range"):
            PhysicsModelParameters(temperature_ref=200.0)


class TestPhysicsBasedModel:
    """Test suite for PhysicsBasedModel class."""
    
    def test_init_default(self):
        """Test default initialization."""
        model = PhysicsBasedModel()
        
        assert model.params.k == 0.13
        assert model.params.temperature_ref == 25.0
        assert not model.is_fitted
        assert model.fit_metrics == {}
    
    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {
            "physics_k": 0.15,
            "temperature_ref": 30.0
        }
        
        model = PhysicsBasedModel(config=config)
        
        assert model.params.k == 0.15
        assert model.params.temperature_ref == 30.0
    
    def test_fit_success(self, sample_battery_data):
        """Test successful model fitting."""
        model = PhysicsBasedModel()
        
        metrics = model.fit(sample_battery_data, target_column="Capacity")
        
        assert model.is_fitted
        assert model.params.initial_capacity is not None
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert metrics["n_samples"] == len(sample_battery_data)
    
    def test_fit_missing_columns(self):
        """Test fit with missing required columns."""
        model = PhysicsBasedModel()
        
        # Missing required columns
        invalid_data = pd.DataFrame({
            'wrong_column': [1, 2, 3]
        })
        
        with pytest.raises(ModelError, match="Missing required columns"):
            model.fit(invalid_data)
    
    def test_predict_not_fitted(self, sample_battery_data):
        """Test prediction with unfitted model."""
        model = PhysicsBasedModel()
        
        with pytest.raises(ModelError, match="Model must be fitted before making predictions"):
            model.predict(sample_battery_data)
    
    def test_predict_success(self, sample_battery_data):
        """Test successful prediction."""
        model = PhysicsBasedModel()
        model.fit(sample_battery_data)
        
        predictions = model.predict(sample_battery_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_battery_data)
        assert np.all(predictions > 0)  # Capacity should be positive
        assert np.all(predictions <= model.params.initial_capacity)  # Should not exceed initial
    
    def test_degradation_factor_calculation(self, sample_battery_data):
        """Test degradation factor calculation."""
        model = PhysicsBasedModel()
        model.fit(sample_battery_data)
        
        degradation_factors = model.get_degradation_factor(sample_battery_data)
        
        assert isinstance(degradation_factors, np.ndarray)
        assert len(degradation_factors) == len(sample_battery_data)
        assert np.all(degradation_factors >= 0)  # Should be non-negative
    
    def test_predict_lifetime(self, sample_battery_data):
        """Test battery lifetime prediction."""
        model = PhysicsBasedModel()
        model.fit(sample_battery_data)
        
        lifetime_info = model.predict_lifetime(
            max_cycles=500,
            temperature=25.0,
            charge_time=3600.0,
            capacity_threshold=0.8
        )
        
        assert isinstance(lifetime_info, dict)
        assert "end_of_life_cycle" in lifetime_info
        assert "end_of_life_capacity" in lifetime_info
        assert "capacity_threshold" in lifetime_info
        assert lifetime_info["capacity_threshold"] == 0.8
        assert lifetime_info["end_of_life_cycle"] > 0
    
    def test_sensitivity_analysis(self, sample_battery_data):
        """Test sensitivity analysis functionality."""
        model = PhysicsBasedModel()
        model.fit(sample_battery_data)
        
        parameter_ranges = {
            'k': (0.1, 0.2)
        }
        
        results = model.sensitivity_analysis(sample_battery_data, parameter_ranges)
        
        assert isinstance(results, dict)
        assert 'k' in results
        assert 'values' in results['k']
        assert 'predictions' in results['k']
        assert 'rmse_vs_base' in results['k']
    
    def test_export_import_parameters(self, sample_battery_data):
        """Test parameter export and import."""
        model = PhysicsBasedModel()
        model.fit(sample_battery_data)
        
        # Export parameters
        exported_params = model.export_parameters()
        
        assert isinstance(exported_params, dict)
        assert "k" in exported_params
        assert "initial_capacity" in exported_params
        assert "is_fitted" in exported_params
        
        # Create new model and import parameters
        new_model = PhysicsBasedModel()
        new_model.import_parameters(exported_params)
        
        assert new_model.is_fitted
        assert new_model.params.k == model.params.k
        assert new_model.params.initial_capacity == model.params.initial_capacity
    
    def test_physics_predictions_realistic(self, sample_battery_data):
        """Test that physics predictions are physically realistic."""
        model = PhysicsBasedModel()
        model.fit(sample_battery_data)
        
        predictions = model.predict(sample_battery_data)
        
        # Physics constraints
        assert np.all(predictions > 0), "Capacity should be positive"
        assert np.all(predictions <= model.params.initial_capacity), "Capacity should not exceed initial"
        
        # Should show degradation over cycles
        cycles = sample_battery_data['id_cycle'].values
        sorted_indices = np.argsort(cycles)
        sorted_predictions = predictions[sorted_indices]
        
        # General trend should be decreasing (allowing for some noise)
        trend_slope = np.polyfit(cycles[sorted_indices], sorted_predictions, 1)[0]
        assert trend_slope <= 0, "Overall trend should be decreasing capacity"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        model = PhysicsBasedModel()
        
        # Edge case: zero or negative time values
        edge_data = pd.DataFrame({
            'id_cycle': [1, 2, 3],
            'Temperature_measured': [25.0, 25.0, 25.0],
            'Time': [0, -100, 3600],  # Invalid time values
            'Capacity': [2.0, 1.9, 1.8],
            'Voltage_measured': [3.7, 3.6, 3.5],
            'Current_measured': [-2.0, -2.0, -2.0]
        })
        
        model.fit(edge_data)
        predictions = model.predict(edge_data)
        
        # Should handle edge cases gracefully
        assert len(predictions) == len(edge_data)
        assert np.all(np.isfinite(predictions))


@pytest.mark.parametrize("k_value,expected_trend", [
    (0.05, "slower_degradation"),
    (0.13, "normal_degradation"), 
    (0.25, "faster_degradation")
])
def test_degradation_coefficient_impact(sample_battery_data, k_value, expected_trend):
    """Test impact of different degradation coefficients."""
    config = {"physics_k": k_value}
    model = PhysicsBasedModel(config=config)
    model.fit(sample_battery_data)
    
    predictions = model.predict(sample_battery_data)
    
    # Higher k should lead to more degradation
    final_capacity_ratio = predictions[-1] / predictions[0]
    
    if expected_trend == "slower_degradation":
        assert final_capacity_ratio > 0.85  # Less degradation
    elif expected_trend == "faster_degradation":
        assert final_capacity_ratio < 0.75  # More degradation
    else:  # normal_degradation
        assert 0.75 <= final_capacity_ratio <= 0.85