"""
Machine Learning correction model for hybrid digital twin.

This module implements the ML component that learns to correct physics model
predictions by modeling the residual between physics predictions and observations.
"""

from typing import Dict, List, Optional, Tuple, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger

from hybrid_digital_twin.utils.exceptions import ModelError, InvalidParameterError


@dataclass
class MLModelConfig:
    """Configuration for the ML correction model."""

    # Architecture
    hidden_layers: List[int] = None
    dropout_rate: float = 0.1
    activation: str = "relu"
    output_activation: Optional[str] = None

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5

    # Regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 0.001

    # Optimization
    optimizer: str = "adam"
    loss_function: str = "mse"

    def __post_init__(self):
        """Set default values and validate configuration."""
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64]

        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise InvalidParameterError("Dropout rate must be between 0 and 1")

        if self.learning_rate <= 0:
            raise InvalidParameterError("Learning rate must be positive")

        if self.batch_size <= 0:
            raise InvalidParameterError("Batch size must be positive")


class MLCorrectionModel:
    """
    Machine Learning model for correcting physics-based predictions.

    This model learns the residual function:
    ΔC = f_ML(C_physics, T, cycle, time, ...)

    Where:
    - ΔC is the correction to be added to physics predictions
    - C_physics is the physics-based prediction
    - Additional features include temperature, cycle number, etc.

    The final hybrid prediction becomes:
    C_hybrid = C_physics + ΔC

    The model uses a deep neural network with configurable architecture,
    regularization, and training procedures optimized for time series
    regression tasks.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Initialize the ML correction model.

        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = MLModelConfig(**({} if config is None else config.get("ml_model", {})))
        self.model: Optional[keras.Model] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history: Dict = {}
        self.feature_names: List[str] = []

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        logger.debug(f"Initialized MLCorrectionModel with config: {self.config}")

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            feature_names: Optional[List[str]] = None,
            **kwargs) -> Dict[str, float]:
        """
        Train the ML correction model.

        Args:
            X: Input features (physics predictions + additional features)
            y: Target residuals (actual - physics predictions)
            validation_data: Optional validation data tuple (X_val, y_val)
            feature_names: Names of input features for interpretability
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training metrics

        Raises:
            ModelError: If training fails
        """
        try:
            logger.info("Training ML correction model")

            # Store feature names
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Handle validation data scaling
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val_scaled = self.scaler.transform(X_val)
                validation_data_scaled = (X_val_scaled, y_val)
            else:
                validation_data_scaled = None

            # Build model architecture
            self.model = self._build_model(input_dim=X_scaled.shape[1])

            # Set up callbacks
            callbacks_list = self._setup_callbacks()

            # Train the model
            history = self.model.fit(
                X_scaled, y,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=validation_data_scaled,
                callbacks=callbacks_list,
                verbose=1,
                **kwargs
            )

            # Store training history
            self.training_history = {
                "loss": history.history["loss"],
                "val_loss": history.history.get("val_loss", []),
                "mae": history.history.get("mae", []),
                "val_mae": history.history.get("val_mae", []),
            }

            # Calculate final metrics
            train_pred = self.model.predict(X_scaled, verbose=0)
            training_metrics = self._calculate_metrics(y, train_pred.flatten())

            if validation_data_scaled is not None:
                val_pred = self.model.predict(X_val_scaled, verbose=0)
                val_metrics = self._calculate_metrics(y_val, val_pred.flatten())
                training_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            self.is_fitted = True
            logger.success(f"ML model training completed. Train RMSE: {training_metrics['rmse']:.4f}")

            return training_metrics

        except Exception as e:
            logger.error(f"ML model training failed: {str(e)}")
            raise ModelError(f"ML model training failed: {str(e)}") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ML correction predictions.

        Args:
            X: Input features

        Returns:
            Predicted corrections (residuals)

        Raises:
            ModelError: If model is not fitted
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before making predictions")

        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled, verbose=0)
            return predictions.flatten()
        except Exception as e:
            logger.error(f"ML prediction failed: {str(e)}")
            raise ModelError(f"ML prediction failed: {str(e)}") from e

    def predict_with_uncertainty(self,
                                X: np.ndarray,
                                n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using Monte Carlo Dropout.

        Args:
            X: Input features
            n_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)

        # Enable dropout during inference for uncertainty estimation
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X_scaled, training=True)  # Enable dropout
            predictions.append(pred.numpy().flatten())

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)

        return mean_pred, uncertainty

    def _build_model(self, input_dim: int) -> keras.Model:
        """Build the neural network architecture."""
        inputs = keras.Input(shape=(input_dim,), name="features")

        x = inputs

        # Add hidden layers
        for i, units in enumerate(self.config.hidden_layers):
            x = layers.Dense(
                units,
                activation=self.config.activation,
                kernel_regularizer=keras.regularizers.L1L2(
                    l1=self.config.l1_regularization,
                    l2=self.config.l2_regularization
                ),
                name=f"dense_{i+1}"
            )(x)

            if self.config.dropout_rate > 0:
                x = layers.Dropout(
                    self.config.dropout_rate,
                    name=f"dropout_{i+1}"
                )(x)

        # Output layer
        outputs = layers.Dense(
            1,
            activation=self.config.output_activation,
            name="output"
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="ml_correction_model")

        # Compile model
        optimizer = self._get_optimizer()
        model.compile(
            optimizer=optimizer,
            loss=self.config.loss_function,
            metrics=["mae", "mse"]
        )

        logger.debug(f"Built model with {model.count_params()} parameters")
        return model

    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Get the configured optimizer."""
        if self.config.optimizer.lower() == "adam":
            return optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == "sgd":
            return optimizers.SGD(learning_rate=self.config.learning_rate, momentum=0.9)
        elif self.config.optimizer.lower() == "rmsprop":
            return optimizers.RMSprop(learning_rate=self.config.learning_rate)
        else:
            raise InvalidParameterError(f"Unknown optimizer: {self.config.optimizer}")

    def _setup_callbacks(self) -> List[keras.callbacks.Callback]:
        """Set up training callbacks."""
        callbacks_list = []

        # Early stopping
        if self.config.early_stopping_patience > 0:
            early_stopping = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks_list.append(early_stopping)

        # Learning rate reduction
        if self.config.reduce_lr_patience > 0:
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
            callbacks_list.append(reduce_lr)

        return callbacks_list

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        max_error = np.max(np.abs(y_true - y_pred))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "max_error": max_error,
        }

    def get_feature_importance(self, X: np.ndarray, method: str = "permutation") -> Dict[str, float]:
        """
        Calculate feature importance using the specified method.

        Args:
            X: Input features for importance calculation
            method: Method to use ("permutation", "gradient")

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ModelError("Model must be fitted before calculating feature importance")

        if method == "permutation":
            return self._permutation_importance(X)
        elif method == "gradient":
            return self._gradient_importance(X)
        else:
            raise InvalidParameterError(f"Unknown importance method: {method}")

    def _permutation_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate permutation-based feature importance."""
        baseline_pred = self.predict(X)
        baseline_score = np.mean(baseline_pred ** 2)  # MSE as baseline

        importance_scores = {}

        for i, feature_name in enumerate(self.feature_names):
            # Permute the i-th feature
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

            # Calculate score with permuted feature
            permuted_pred = self.predict(X_permuted)
            permuted_score = np.mean(permuted_pred ** 2)

            # Importance is the increase in error
            importance_scores[feature_name] = permuted_score - baseline_score

        return importance_scores

    def _gradient_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate gradient-based feature importance."""
        X_scaled = self.scaler.transform(X)
        X_tensor = tf.constant(X_scaled, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
            loss = tf.reduce_mean(tf.square(predictions))

        gradients = tape.gradient(loss, X_tensor)
        importance_scores = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()

        return dict(zip(self.feature_names, importance_scores))

    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save the model and associated components."""
        if not self.is_fitted:
            raise ModelError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(filepath / "keras_model")

        # Save scaler and metadata
        import joblib
        joblib.dump(self.scaler, filepath / "scaler.joblib")

        metadata = {
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "training_history": self.training_history,
        }

        with open(filepath / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"ML model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "MLCorrectionModel":
        """Load a saved model."""
        filepath = Path(filepath)

        # Load metadata
        with open(filepath / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Create instance
        instance = cls({"ml_model": metadata["config"]})

        # Load components
        instance.model = keras.models.load_model(filepath / "keras_model")

        import joblib
        instance.scaler = joblib.load(filepath / "scaler.joblib")

        instance.feature_names = metadata["feature_names"]
        instance.training_history = metadata["training_history"]
        instance.is_fitted = True

        logger.info(f"ML model loaded from {filepath}")
        return instance
