"""
Unit tests for the HybridDigitalTwin class.

This module contains comprehensive unit tests for the core digital twin
functionality, focusing on individual component behavior and integration.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import joblib

from hybrid_digital_twin.core.digital_twin import HybridDigitalTwin, PredictionResult
from hybrid_digital_twin.models.physics_model import PhysicsBasedModel
from hybrid_digital_twin.models.ml_model import MLCorrectionModel
from hybrid_digital_twin.utils.exceptions import (
    DigitalTwinError,
    ModelNotTrainedError,
    InvalidDataError,
)


class TestHybridDigitalTwin:
    """Test suite for HybridDigitalTwin class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample battery data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        cycles = np.arange(1, n_samples + 1)
        temperature = 25.0 + np.random.normal(0, 2, n_samples)
        time = 3600 + np.random.normal(0, 100, n_samples)
        
        # Simulate capacity degradation
        initial_capacity = 2.0
        degradation = 0.0001 * cycles * temperature / time
        capacity = initial_capacity * np.exp(-degradation)
        capacity += np.random.normal(0, 0.001, n_samples)  # Add noise
        
        return pd.DataFrame({
            'id_cycle': cycles,
            'Temperature_measured': temperature,
            'Time': time,
            'Capacity': capacity,
            'Voltage_measured': 3.5 + np.random.normal(0, 0.1, n_samples),
            'Current_measured': -2.0 + np.random.normal(0, 0.05, n_samples),
        })
    
    @pytest.fixture
    def mock_physics_model(self):
        """Create a mock physics model."""
        model = Mock(spec=PhysicsBasedModel)
        model.fit.return_value = {'rmse': 0.01, 'mae': 0.008, 'r2': 0.98}
        model.predict.return_value = np.array([1.98, 1.97, 1.96, 1.95])
        model.is_fitted = True
        return model
    
    @pytest.fixture
    def mock_ml_model(self):
        """Create a mock ML model."""
        model = Mock(spec=MLCorrectionModel)
        model.fit.return_value = {'rmse': 0.005, 'mae': 0.004, 'r2': 0.99}
        model.predict.return_value = np.array([0.001, -0.002, 0.001, 0.0])
        model.is_fitted = True
        return model
    
    def test_init_default(self):
        """Test default initialization."""
        twin = HybridDigitalTwin()
        
        assert twin.physics_model is not None
        assert twin.ml_model is not None
        assert not twin.is_trained
        assert twin.metrics is not None
        assert twin.training_history == {}
    
    def test_init_with_models(self, mock_physics_model, mock_ml_model):
        """Test initialization with provided models."""
        config = {'test': 'value'}
        
        twin = HybridDigitalTwin(
            physics_model=mock_physics_model,
            ml_model=mock_ml_model,
            config=config
        )
        
        assert twin.physics_model is mock_physics_model
        assert twin.ml_model is mock_ml_model
        assert twin.config == config
    
    def test_fit_success(self, sample_data, mock_physics_model, mock_ml_model):
        """Test successful model training."""
        twin = HybridDigitalTwin(
            physics_model=mock_physics_model,
            ml_model=mock_ml_model
        )
        
        # Mock the _extract_ml_features method
        with patch.object(twin, '_extract_ml_features') as mock_extract:
            mock_extract.return_value = np.random.rand(len(sample_data), 3)
            
            metrics = twin.fit(sample_data, validation_split=0.2)
        
        # Verify models were called
        mock_physics_model.fit.assert_called_once()
        mock_ml_model.fit.assert_called_once()
        
        # Verify training state
        assert twin.is_trained
        assert isinstance(metrics, dict)
        assert 'train_rmse' in metrics or 'hybrid_metrics' in twin.training_history
    
    def test_fit_invalid_data(self):
        """Test fit with invalid data."""
        twin = HybridDigitalTwin()
        
        # Empty DataFrame
        empty_data = pd.DataFrame()
        with pytest.raises(InvalidDataError):
            twin.fit(empty_data)
        
        # Missing required columns
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        with pytest.raises(InvalidDataError):
            twin.fit(invalid_data)
    
    def test_predict_not_trained(self, sample_data):
        """Test prediction with untrained model."""
        twin = HybridDigitalTwin()
        
        with pytest.raises(ModelNotTrainedError):
            twin.predict(sample_data)
    
    def test_predict_success(self, sample_data, mock_physics_model, mock_ml_model):
        """Test successful prediction."""
        twin = HybridDigitalTwin(
            physics_model=mock_physics_model,
            ml_model=mock_ml_model
        )
        twin.is_trained = True
        
        # Mock feature extraction
        with patch.object(twin, '_extract_ml_features') as mock_extract:
            mock_extract.return_value = np.random.rand(len(sample_data), 3)
            
            predictions = twin.predict(sample_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_data)
        
        # Verify model calls
        mock_physics_model.predict.assert_called_once()
        mock_ml_model.predict.assert_called_once()
    
    def test_predict_with_components(self, sample_data, mock_physics_model, mock_ml_model):
        """Test prediction returning components."""
        twin = HybridDigitalTwin(
            physics_model=mock_physics_model,
            ml_model=mock_ml_model
        )
        twin.is_trained = True
        
        with patch.object(twin, '_extract_ml_features') as mock_extract:
            mock_extract.return_value = np.random.rand(len(sample_data), 3)
            
            result = twin.predict(sample_data, return_components=True)
        
        assert isinstance(result, PredictionResult)
        assert result.physics_prediction is not None
        assert result.ml_correction is not None
        assert result.hybrid_prediction is not None
        assert result.metadata is not None
    
    def test_predict_future(self, mock_physics_model, mock_ml_model):
        """Test future prediction functionality."""
        twin = HybridDigitalTwin(
            physics_model=mock_physics_model,
            ml_model=mock_ml_model
        )
        twin.is_trained = True
        
        cycles = np.array([200, 250, 300])
        temperature = 25.0
        charge_time = 3600.0
        initial_capacity = 2.0
        
        with patch.object(twin, 'predict') as mock_predict:
            mock_predict.return_value = PredictionResult(
                physics_prediction=np.array([1.8, 1.7, 1.6]),
                ml_correction=np.array([0.01, 0.02, 0.01]),
                hybrid_prediction=np.array([1.81, 1.72, 1.61]),
            )
            
            result = twin.predict_future(
                cycles=cycles,
                temperature=temperature,
                charge_time=charge_time,
                initial_capacity=initial_capacity,
                return_uncertainty=True
            )
        
        assert isinstance(result, PredictionResult)
        mock_predict.assert_called_once()
    
    def test_evaluate(self, sample_data, mock_physics_model, mock_ml_model):
        """Test model evaluation."""
        twin = HybridDigitalTwin(
            physics_model=mock_physics_model,
            ml_model=mock_ml_model
        )
        twin.is_trained = True
        
        # Mock predict method
        with patch.object(twin, 'predict') as mock_predict:
            mock_predict.return_value = sample_data['Capacity'].values + np.random.normal(0, 0.01, len(sample_data))
            
            metrics = twin.evaluate(sample_data)
        
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
    
    def test_save_load_model(self, sample_data, mock_physics_model, mock_ml_model):
        """Test model saving and loading."""
        twin = HybridDigitalTwin(
            physics_model=mock_physics_model,
            ml_model=mock_ml_model
        )
        twin.is_trained = True
        twin.training_history = {'test': 'data'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            
            # Save model
            twin.save_model(model_path)
            assert model_path.exists()
            
            # Load model
            loaded_twin = HybridDigitalTwin.load_model(model_path)
            assert loaded_twin.is_trained
            assert loaded_twin.training_history == twin.training_history
    
    def test_save_untrained_model(self):
        """Test saving untrained model raises error."""
        twin = HybridDigitalTwin()
        
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ModelNotTrainedError):
                twin.save_model(temp_file.name)
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model file."""
        with pytest.raises(FileNotFoundError):
            HybridDigitalTwin.load_model("nonexistent_file.pkl")
    
    def test_extract_ml_features(self, sample_data):
        """Test ML feature extraction."""
        twin = HybridDigitalTwin()
        physics_pred = np.random.rand(len(sample_data))
        
        features = twin._extract_ml_features(sample_data, physics_pred)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_data)
        assert features.shape[1] > 1  # Should have multiple features
    
    def test_estimate_uncertainty(self):
        """Test uncertainty estimation."""
        twin = HybridDigitalTwin()
        features = np.random.rand(100, 5)
        
        uncertainty = twin._estimate_uncertainty(features)
        
        assert isinstance(uncertainty, np.ndarray)
        assert len(uncertainty) == len(features)
        assert np.all(uncertainty >= 0)  # Uncertainty should be non-negative
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        twin = HybridDigitalTwin()
        
        y_train_true = np.array([1.0, 1.1, 1.2, 1.3])
        y_val_true = np.array([1.4, 1.5])
        y_train_pred = np.array([1.01, 1.09, 1.21, 1.28])
        y_val_pred = np.array([1.42, 1.48])
        
        metrics = twin._calculate_metrics(
            y_train_true, y_val_true, y_train_pred, y_val_pred
        )
        
        assert isinstance(metrics, dict)
        assert 'train_rmse' in metrics
        assert 'val_rmse' in metrics
        assert 'train_mae' in metrics
        assert 'val_mae' in metrics
        assert 'train_r2' in metrics
        assert 'val_r2' in metrics


class TestPredictionResult:
    """Test suite for PredictionResult class."""
    
    def test_prediction_result_creation(self):
        """Test PredictionResult creation."""
        physics_pred = np.array([1.0, 1.1, 1.2])
        ml_correction = np.array([0.01, -0.02, 0.005])
        hybrid_pred = np.array([1.01, 1.08, 1.205])
        uncertainty = np.array([0.005, 0.007, 0.006])
        metadata = {'n_samples': 3}
        
        result = PredictionResult(
            physics_prediction=physics_pred,
            ml_correction=ml_correction,
            hybrid_prediction=hybrid_pred,
            uncertainty=uncertainty,
            metadata=metadata
        )
        
        assert np.array_equal(result.physics_prediction, physics_pred)
        assert np.array_equal(result.ml_correction, ml_correction)
        assert np.array_equal(result.hybrid_prediction, hybrid_pred)
        assert np.array_equal(result.uncertainty, uncertainty)
        assert result.metadata == metadata
    
    def test_prediction_result_optional_fields(self):
        """Test PredictionResult with optional fields."""
        physics_pred = np.array([1.0, 1.1])
        ml_correction = np.array([0.01, -0.02])
        hybrid_pred = np.array([1.01, 1.08])
        
        result = PredictionResult(
            physics_prediction=physics_pred,
            ml_correction=ml_correction,
            hybrid_prediction=hybrid_pred
        )
        
        assert result.uncertainty is None
        assert result.metadata is None


@pytest.fixture
def trained_twin(sample_data):
    """Create a trained hybrid digital twin for testing."""
    twin = HybridDigitalTwin()
    
    # Mock the training process
    with patch.object(twin.physics_model, 'fit') as mock_physics_fit:
        mock_physics_fit.return_value = {'rmse': 0.01}
        with patch.object(twin.physics_model, 'predict') as mock_physics_predict:
            mock_physics_predict.return_value = np.random.rand(len(sample_data))
            with patch.object(twin.ml_model, 'fit') as mock_ml_fit:
                mock_ml_fit.return_value = {'rmse': 0.005}
                with patch.object(twin.ml_model, 'predict') as mock_ml_predict:
                    mock_ml_predict.return_value = np.random.rand(len(sample_data)) * 0.01
                    
                    twin.fit(sample_data)
    
    return twin


class TestHybridDigitalTwinIntegration:
    """Integration tests for the complete hybrid digital twin workflow."""
    
    def test_end_to_end_workflow(self, sample_data):
        """Test complete end-to-end workflow."""
        twin = HybridDigitalTwin()
        
        # Training
        metrics = twin.fit(sample_data, validation_split=0.3)
        assert twin.is_trained
        assert isinstance(metrics, dict)
        
        # Prediction
        predictions = twin.predict(sample_data[:10])
        assert len(predictions) == 10
        
        # Evaluation
        eval_metrics = twin.evaluate(sample_data[10:20])
        assert 'rmse' in eval_metrics
        
        # Future prediction
        future_result = twin.predict_future(
            cycles=np.array([200, 300, 400]),
            temperature=25.0,
            charge_time=3600.0,
            initial_capacity=2.0
        )
        assert isinstance(future_result, PredictionResult)
    
    def test_configuration_impact(self, sample_data):
        """Test that configuration affects model behavior."""
        config1 = {
            'ml_model': {
                'hidden_layers': [32, 16],
                'epochs': 10
            }
        }
        
        config2 = {
            'ml_model': {
                'hidden_layers': [64, 32, 16],
                'epochs': 20
            }
        }
        
        twin1 = HybridDigitalTwin(config=config1)
        twin2 = HybridDigitalTwin(config=config2)
        
        # Both should train successfully but potentially with different performance
        metrics1 = twin1.fit(sample_data)
        metrics2 = twin2.fit(sample_data)
        
        assert isinstance(metrics1, dict)
        assert isinstance(metrics2, dict)
        
        # Models should produce different predictions due to different architectures
        pred1 = twin1.predict(sample_data[:5])
        pred2 = twin2.predict(sample_data[:5])
        
        # Allow for small differences due to randomness
        assert not np.allclose(pred1, pred2, rtol=0.01)