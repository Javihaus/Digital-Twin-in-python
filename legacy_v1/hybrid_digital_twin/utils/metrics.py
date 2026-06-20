"""
Model performance metrics and evaluation utilities.

This module provides comprehensive metrics calculation for evaluating
the performance of both physics-based and machine learning models
in the hybrid digital twin framework.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from hybrid_digital_twin.utils.exceptions import InvalidDataError


@dataclass
class MetricsResult:
    """Container for model performance metrics."""

    rmse: float
    mae: float
    r2: float
    mape: float
    median_ae: float
    max_error: float
    mean_error: float
    std_error: float
    n_samples: int

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
            "median_ae": self.median_ae,
            "max_error": self.max_error,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"MetricsResult(\n"
            f"  RMSE: {self.rmse:.4f}\n"
            f"  MAE: {self.mae:.4f}\n"
            f"  R²: {self.r2:.4f}\n"
            f"  MAPE: {self.mape:.2f}%\n"
            f"  Max Error: {self.max_error:.4f}\n"
            f"  Samples: {self.n_samples}\n"
            f")"
        )


class ModelMetrics:
    """
    Comprehensive model evaluation metrics calculator.

    This class provides methods to calculate various performance metrics
    for regression tasks, with special considerations for battery capacity
    prediction and time series evaluation.
    """

    def __init__(self):
        """Initialize the metrics calculator."""
        pass

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive set of regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            sample_weight: Optional sample weights

        Returns:
            Dictionary containing all calculated metrics

        Raises:
            InvalidDataError: If input arrays are invalid
        """
        self._validate_inputs(y_true, y_pred, sample_weight)

        # Basic regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
        mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
        median_ae = median_absolute_error(y_true, y_pred)

        # Error statistics
        errors = y_true - y_pred
        max_error = np.max(np.abs(errors))
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # Percentage-based metrics (handle division by zero)
        mape = self._safe_mape(y_true, y_pred)

        # Additional metrics
        mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "median_ae": median_ae,
            "max_error": max_error,
            "mean_error": mean_error,
            "std_error": std_error,
            "mse": mse,
            "n_samples": len(y_true),
        }

    def calculate_metrics_result(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        """
        Calculate metrics and return as MetricsResult object.

        Args:
            y_true: True target values
            y_pred: Predicted values
            sample_weight: Optional sample weights

        Returns:
            MetricsResult object containing all metrics
        """
        metrics_dict = self.calculate_all_metrics(y_true, y_pred, sample_weight)

        return MetricsResult(
            rmse=metrics_dict["rmse"],
            mae=metrics_dict["mae"],
            r2=metrics_dict["r2"],
            mape=metrics_dict["mape"],
            median_ae=metrics_dict["median_ae"],
            max_error=metrics_dict["max_error"],
            mean_error=metrics_dict["mean_error"],
            std_error=metrics_dict["std_error"],
            n_samples=metrics_dict["n_samples"],
        )

    def compare_models(
        self,
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.

        Args:
            y_true: True target values
            predictions_dict: Dictionary mapping model names to predictions
            metrics: List of metrics to include in comparison

        Returns:
            DataFrame with models as rows and metrics as columns
        """
        if metrics is None:
            metrics = ["rmse", "mae", "r2", "mape"]

        comparison_results = []

        for model_name, y_pred in predictions_dict.items():
            model_metrics = self.calculate_all_metrics(y_true, y_pred)
            row = {"model": model_name}
            row.update({metric: model_metrics[metric] for metric in metrics})
            comparison_results.append(row)

        df = pd.DataFrame(comparison_results)
        df = df.set_index("model")

        # Rank models (lower is better for error metrics, higher for R²)
        for metric in metrics:
            if metric == "r2":
                df[f"{metric}_rank"] = df[metric].rank(ascending=False)
            else:
                df[f"{metric}_rank"] = df[metric].rank(ascending=True)

        logger.debug(f"Model comparison completed for {len(predictions_dict)} models")
        return df

    def calculate_time_series_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        time_index: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate time series specific metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            time_index: Optional time index for temporal analysis

        Returns:
            Dictionary containing time series metrics
        """
        base_metrics = self.calculate_all_metrics(y_true, y_pred)

        # Time series specific metrics
        ts_metrics = {}

        # Directional accuracy (for trend prediction)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction)
            ts_metrics["directional_accuracy"] = directional_accuracy

        # Forecast bias
        forecast_bias = np.mean(y_pred - y_true)
        ts_metrics["forecast_bias"] = forecast_bias

        # Theil's U statistic (if applicable)
        if len(y_true) > 1:
            naive_forecast = np.roll(y_true, 1)[1:]  # Previous value as forecast
            actual_next = y_true[1:]
            model_next = y_pred[1:]

            naive_mse = mean_squared_error(actual_next, naive_forecast)
            model_mse = mean_squared_error(actual_next, model_next)

            if naive_mse > 0:
                theil_u = np.sqrt(model_mse) / np.sqrt(naive_mse)
                ts_metrics["theil_u"] = theil_u

        # Combine with base metrics
        ts_metrics.update(base_metrics)
        return ts_metrics

    def calculate_residual_statistics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate detailed residual statistics for model diagnostics.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary containing residual statistics
        """
        residuals = y_true - y_pred

        stats_dict = {
            "residual_mean": np.mean(residuals),
            "residual_std": np.std(residuals),
            "residual_min": np.min(residuals),
            "residual_max": np.max(residuals),
            "residual_q25": np.percentile(residuals, 25),
            "residual_q50": np.percentile(residuals, 50),
            "residual_q75": np.percentile(residuals, 75),
            "residual_iqr": np.percentile(residuals, 75) - np.percentile(residuals, 25),
        }

        # Normality test (Shapiro-Wilk for small samples, Kolmogorov-Smirnov for large)
        if len(residuals) <= 5000:
            stat, p_value = stats.shapiro(residuals)
            stats_dict["normality_test"] = "shapiro"
        else:
            stat, p_value = stats.kstest(residuals, "norm")
            stats_dict["normality_test"] = "ks"

        stats_dict["normality_statistic"] = stat
        stats_dict["normality_p_value"] = p_value
        stats_dict["is_normal"] = p_value > 0.05

        # Durbin-Watson test for autocorrelation (if applicable)
        if len(residuals) > 10:
            dw_stat = self._durbin_watson(residuals)
            stats_dict["durbin_watson"] = dw_stat

        return stats_dict

    def calculate_confidence_intervals(
        self, y_true: np.ndarray, y_pred: np.ndarray, confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for model metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            confidence_level: Confidence level (default 0.95)

        Returns:
            Dictionary mapping metric names to (lower_bound, upper_bound) tuples
        """
        n_samples = len(y_true)
        alpha = 1 - confidence_level

        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_metrics = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            metrics = self.calculate_all_metrics(y_true_boot, y_pred_boot)
            bootstrap_metrics.append(metrics)

        # Calculate confidence intervals
        confidence_intervals = {}
        bootstrap_df = pd.DataFrame(bootstrap_metrics)

        for metric in bootstrap_df.columns:
            if metric in ["n_samples"]:  # Skip non-metric columns
                continue

            lower = np.percentile(bootstrap_df[metric], (alpha / 2) * 100)
            upper = np.percentile(bootstrap_df[metric], (1 - alpha / 2) * 100)
            confidence_intervals[metric] = (lower, upper)

        return confidence_intervals

    def _validate_inputs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Validate input arrays for metrics calculation."""
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)

        if len(y_true) != len(y_pred):
            raise InvalidDataError(
                f"y_true and y_pred must have same length: {len(y_true)} != {len(y_pred)}"
            )

        if len(y_true) == 0:
            raise InvalidDataError("Input arrays cannot be empty")

        if sample_weight is not None:
            if len(sample_weight) != len(y_true):
                raise InvalidDataError(
                    f"sample_weight length {len(sample_weight)} != {len(y_true)}"
                )

        if not np.all(np.isfinite(y_true)):
            raise InvalidDataError("y_true contains non-finite values")

        if not np.all(np.isfinite(y_pred)):
            raise InvalidDataError("y_pred contains non-finite values")

    def _safe_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE with handling for zero values."""
        # Use symmetric MAPE for better handling of zero values
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator > 1e-8  # Avoid division by very small numbers

        if not np.any(mask):
            return np.inf

        mape_values = np.abs(y_true - y_pred) / denominator
        return np.mean(mape_values[mask]) * 100

    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson test statistic for autocorrelation."""
        diff_residuals = np.diff(residuals)
        return np.sum(diff_residuals**2) / np.sum(residuals**2)

    def export_metrics_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        include_residuals: bool = True,
        include_confidence: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model for the report
            include_residuals: Whether to include residual statistics
            include_confidence: Whether to include confidence intervals

        Returns:
            Dictionary containing complete metrics report
        """
        report = {
            "model_name": model_name,
            "basic_metrics": self.calculate_all_metrics(y_true, y_pred),
            "time_series_metrics": self.calculate_time_series_metrics(y_true, y_pred),
        }

        if include_residuals:
            report["residual_statistics"] = self.calculate_residual_statistics(
                y_true, y_pred
            )

        if include_confidence:
            report["confidence_intervals"] = self.calculate_confidence_intervals(
                y_true, y_pred
            )

        logger.info(f"Generated comprehensive metrics report for {model_name}")
        return report
