"""
End-to-end integration tests for the hybrid digital twin framework.

This module contains integration tests that verify the complete workflow
from data loading to model training and prediction.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from hybrid_digital_twin import HybridDigitalTwin, BatteryDataLoader
from hybrid_digital_twin.core.digital_twin import PredictionResult


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow_minimal_data(self, minimal_data, temp_dir):
        """Test complete workflow with minimal data."""
        # Save test data
        data_file = temp_dir / "test_data.csv" 
        minimal_data.to_csv(data_file, index=False)
        
        # Load data
        loader = BatteryDataLoader(data_dir=temp_dir)
        data = loader.load_csv(data_file.name)
        
        assert len(data) == len(minimal_data)
        
        # Train model with minimal configuration
        config = {
            "ml_model": {
                "hidden_layers": [4],
                "epochs": 2,
                "batch_size": 2
            }
        }
        
        twin = HybridDigitalTwin(config=config)
        
        # Training
        metrics = twin.fit(data, validation_split=0.4)  # Large validation split for small data
        
        assert twin.is_trained
        assert "train_rmse" in metrics or len(twin.training_history) > 0
        
        # Prediction
        predictions = twin.predict(data)
        assert len(predictions) == len(data)
        
        # Save and load model
        model_file = temp_dir / "test_model.pkl"
        twin.save_model(model_file)
        
        loaded_twin = HybridDigitalTwin.load_model(model_file)
        assert loaded_twin.is_trained
        
        # Test loaded model predictions
        loaded_predictions = loaded_twin.predict(data)
        assert len(loaded_predictions) == len(data)
        
        # Predictions should be close (allowing for small numerical differences)
        np.testing.assert_allclose(predictions, loaded_predictions, rtol=1e-5)
    
    @pytest.mark.slow
    def test_realistic_battery_workflow(self, sample_battery_data, temp_dir):
        """Test workflow with realistic battery data."""
        # Split data
        train_size = int(0.7 * len(sample_battery_data))
        train_data = sample_battery_data.iloc[:train_size].copy()
        test_data = sample_battery_data.iloc[train_size:].copy()
        
        # Train model
        config = {
            "physics_k": 0.13,
            "ml_model": {
                "hidden_layers": [16, 8],
                "epochs": 10,
                "batch_size": 8
            }
        }
        
        twin = HybridDigitalTwin(config=config)
        metrics = twin.fit(train_data, validation_split=0.2)
        
        # Verify training
        assert twin.is_trained
        assert isinstance(metrics, dict)
        
        # Test prediction with components
        results = twin.predict(test_data, return_components=True)
        
        assert isinstance(results, PredictionResult)
        assert len(results.physics_prediction) == len(test_data)
        assert len(results.ml_correction) == len(test_data)
        assert len(results.hybrid_prediction) == len(test_data)
        
        # Evaluate performance
        eval_metrics = twin.evaluate(test_data)
        assert "rmse" in eval_metrics
        assert "mae" in eval_metrics
        assert "r2" in eval_metrics
        
        # Verify reasonable performance
        assert eval_metrics["rmse"] < 0.5  # Should be reasonable for test data
        assert eval_metrics["r2"] > 0.0    # Should have some predictive power
        
        # Test future prediction
        future_cycles = np.array([150, 200, 250])
        future_results = twin.predict_future(
            cycles=future_cycles,
            temperature=25.0,
            charge_time=3600.0,
            initial_capacity=2.0
        )
        
        assert isinstance(future_results, PredictionResult)
        assert len(future_results.hybrid_prediction) == len(future_cycles)


@pytest.mark.integration 
class TestDataLoaderIntegration:
    """Integration tests for data loading functionality."""
    
    def test_csv_loading_and_processing(self, sample_battery_data, temp_dir):
        """Test CSV loading and processing pipeline."""
        # Save data with multiple batteries
        multi_battery_data = sample_battery_data.copy()
        multi_battery_data.loc[:30, 'Battery'] = 'B0006'
        multi_battery_data.loc[31:60, 'Battery'] = 'B0007'
        multi_battery_data.loc[61:, 'Battery'] = 'B0005'
        
        data_file = temp_dir / "multi_battery.csv"
        multi_battery_data.to_csv(data_file, index=False)
        
        loader = BatteryDataLoader(data_dir=temp_dir)
        
        # Test single battery loading
        single_battery = loader.load_csv(data_file.name, battery_filter="B0005")
        assert len(single_battery) == len(multi_battery_data[multi_battery_data['Battery'] == 'B0005'])
        
        # Test multiple battery loading
        multiple_batteries = loader.load_multiple_batteries(
            data_file.name,
            battery_ids=["B0005", "B0006"],
            combine=True
        )
        
        assert len(multiple_batteries) == len(multi_battery_data[
            multi_battery_data['Battery'].isin(["B0005", "B0006"])
        ])
        
        # Test data preprocessing
        processed_data, feature_names = loader.preprocess_for_modeling(
            single_battery,
            target_column="Capacity",
            add_derived_features=True
        )
        
        assert len(feature_names) > len(single_battery.columns) - 1  # Should add features
        assert "Capacity" not in feature_names  # Target should not be in features


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration management."""
    
    def test_different_configurations(self, minimal_data):
        """Test models with different configurations."""
        configs = [
            {
                "physics_k": 0.1,
                "ml_model": {"hidden_layers": [4], "epochs": 2}
            },
            {
                "physics_k": 0.15,
                "ml_model": {"hidden_layers": [8, 4], "epochs": 3}
            }
        ]
        
        models = []
        for i, config in enumerate(configs):
            twin = HybridDigitalTwin(config=config)
            
            # Train model
            metrics = twin.fit(minimal_data, validation_split=0.4)
            models.append(twin)
            
            assert twin.is_trained
            assert isinstance(metrics, dict)
        
        # Test that different configs produce different results
        pred1 = models[0].predict(minimal_data)
        pred2 = models[1].predict(minimal_data)
        
        # Should be different (allowing for some tolerance due to randomness)
        assert not np.allclose(pred1, pred2, rtol=0.1)


@pytest.mark.integration
@pytest.mark.slow
def test_performance_benchmark(sample_battery_data):
    """Basic performance benchmark test."""
    import time
    
    twin = HybridDigitalTwin(config={
        "ml_model": {
            "hidden_layers": [32, 16],
            "epochs": 20
        }
    })
    
    # Training time
    start_time = time.time()
    twin.fit(sample_battery_data, validation_split=0.2)
    training_time = time.time() - start_time
    
    # Prediction time  
    start_time = time.time()
    predictions = twin.predict(sample_battery_data)
    prediction_time = time.time() - start_time
    
    # Basic performance expectations
    assert training_time < 300  # Should train in less than 5 minutes
    assert prediction_time < 10  # Should predict in less than 10 seconds
    assert len(predictions) == len(sample_battery_data)