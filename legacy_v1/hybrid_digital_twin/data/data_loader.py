"""
Battery data loading and preprocessing utilities.

This module provides comprehensive data loading capabilities for various
battery datasets commonly used in digital twin applications.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from hybrid_digital_twin.utils.exceptions import DataLoaderError, InvalidDataError
from hybrid_digital_twin.utils.validators import (
    sanitize_numeric_data,
    validate_battery_data,
)


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""

    name: str
    n_samples: int
    n_features: int
    capacity_range: Tuple[float, float]
    cycle_range: Tuple[int, int]
    temperature_range: Tuple[float, float]
    batteries: List[str]

    def __str__(self) -> str:
        return (
            f"Dataset: {self.name}\n"
            f"Samples: {self.n_samples:,}\n"
            f"Features: {self.n_features}\n"
            f"Capacity Range: {self.capacity_range[0]:.3f} - {self.capacity_range[1]:.3f} Ah\n"
            f"Cycle Range: {self.cycle_range[0]} - {self.cycle_range[1]}\n"
            f"Temperature Range: {self.temperature_range[0]:.1f} - {self.temperature_range[1]:.1f} °C\n"
            f"Batteries: {', '.join(self.batteries)}"
        )


class BatteryDataLoader:
    """
    Professional data loader for battery datasets.

    This class provides methods to load, validate, and preprocess various
    battery dataset formats commonly used in digital twin applications.

    Supported formats:
    - NASA Battery Dataset
    - Custom CSV formats
    - HDF5 files
    - Excel files

    Features:
    - Automatic data validation
    - Missing value handling
    - Outlier detection and correction
    - Feature engineering
    - Data quality reporting
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the data loader.

        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.loaded_datasets: Dict[str, pd.DataFrame] = {}

        logger.debug(f"Initialized BatteryDataLoader with data_dir: {self.data_dir}")

    def load_csv(
        self,
        filepath: Union[str, Path],
        battery_filter: Optional[str] = None,
        validate: bool = True,
        clean: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load battery data from CSV file.

        Args:
            filepath: Path to CSV file
            battery_filter: Filter to specific battery ID
            validate: Whether to validate data after loading
            clean: Whether to clean and sanitize data
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded and processed DataFrame

        Raises:
            DataLoaderError: If file loading fails
            InvalidDataError: If data validation fails
        """
        try:
            filepath = self._resolve_path(filepath)
            logger.info(f"Loading CSV data from {filepath}")

            # Load CSV with error handling
            try:
                data = pd.read_csv(filepath, **kwargs)
            except Exception as e:
                raise DataLoaderError(f"Failed to read CSV file: {str(e)}") from e

            if data.empty:
                raise DataLoaderError("Loaded CSV file is empty")

            logger.debug(f"Loaded {len(data):,} rows and {len(data.columns)} columns")

            # Filter by battery if requested
            if battery_filter and "Battery" in data.columns:
                original_size = len(data)
                data = data[data["Battery"] == battery_filter].copy()
                logger.debug(
                    f"Filtered to battery {battery_filter}: {len(data):,} rows (from {original_size:,})"
                )

            # Clean data if requested
            if clean:
                data = self._clean_data(data)

            # Validate data if requested
            if validate:
                validate_battery_data(data)

            # Store in cache
            cache_key = f"{filepath.stem}_{battery_filter or 'all'}"
            self.loaded_datasets[cache_key] = data

            logger.success(f"Successfully loaded and processed data: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Failed to load CSV data: {str(e)}")
            raise DataLoaderError(f"CSV loading failed: {str(e)}") from e

    def load_nasa_dataset(
        self,
        battery_id: str = "B0005",
        dataset_path: Optional[Union[str, Path]] = None,
        temperature_filter: Optional[Tuple[float, float]] = None,
    ) -> pd.DataFrame:
        """
        Load NASA battery dataset with standardized preprocessing.

        Args:
            battery_id: Battery identifier (e.g., "B0005", "B0006")
            dataset_path: Path to NASA dataset file
            temperature_filter: Optional temperature range filter (min, max)

        Returns:
            Processed NASA battery data
        """
        try:
            if dataset_path is None:
                dataset_path = self.data_dir / "raw" / "discharge.csv"

            logger.info(f"Loading NASA dataset for battery {battery_id}")

            # Load data
            data = self.load_csv(dataset_path, battery_filter=battery_id)

            # NASA dataset specific processing
            if "Temperature_measured" in data.columns and temperature_filter:
                min_temp, max_temp = temperature_filter
                original_size = len(data)
                data = data[
                    (data["Temperature_measured"] >= min_temp)
                    & (data["Temperature_measured"] <= max_temp)
                ].copy()
                logger.debug(
                    f"Temperature filter applied: {len(data):,} rows (from {original_size:,})"
                )

            # Group by cycle and get max values (end of discharge)
            if "id_cycle" in data.columns:
                cycle_data = (
                    data.groupby("id_cycle")
                    .agg(
                        {
                            "Capacity": "first",  # Capacity at start of cycle
                            "Temperature_measured": "mean",
                            "Time": "max",  # Total cycle time
                            "Voltage_measured": "mean",
                            "Current_measured": "mean",
                        }
                    )
                    .reset_index()
                )

                # Add cumulative time
                cycle_data["Cumulated_T"] = cycle_data["Time"].cumsum()

                logger.debug(f"Aggregated to {len(cycle_data)} cycles")
                data = cycle_data

            return data

        except Exception as e:
            logger.error(f"Failed to load NASA dataset: {str(e)}")
            raise DataLoaderError(f"NASA dataset loading failed: {str(e)}") from e

    def load_multiple_batteries(
        self,
        filepath: Union[str, Path],
        battery_ids: Optional[List[str]] = None,
        combine: bool = False,
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load data for multiple batteries.

        Args:
            filepath: Path to data file
            battery_ids: List of battery IDs to load
            combine: Whether to combine all batteries into single DataFrame

        Returns:
            Dictionary mapping battery ID to DataFrame, or combined DataFrame
        """
        try:
            filepath = self._resolve_path(filepath)

            # Load full dataset
            full_data = pd.read_csv(filepath)

            if "Battery" not in full_data.columns:
                raise DataLoaderError(
                    "Dataset must contain 'Battery' column for multi-battery loading"
                )

            available_batteries = full_data["Battery"].unique().tolist()

            if battery_ids is None:
                battery_ids = available_batteries
            else:
                # Validate requested battery IDs
                missing_batteries = set(battery_ids) - set(available_batteries)
                if missing_batteries:
                    raise DataLoaderError(f"Batteries not found: {missing_batteries}")

            logger.info(f"Loading {len(battery_ids)} batteries: {battery_ids}")

            battery_data = {}
            for battery_id in battery_ids:
                data = full_data[full_data["Battery"] == battery_id].copy()
                data = self._clean_data(data)
                validate_battery_data(data)
                battery_data[battery_id] = data

                logger.debug(f"Loaded battery {battery_id}: {len(data):,} samples")

            if combine:
                combined_data = pd.concat(battery_data.values(), ignore_index=True)
                logger.info(
                    f"Combined {len(battery_ids)} batteries: {len(combined_data):,} total samples"
                )
                return combined_data

            return battery_data

        except Exception as e:
            logger.error(f"Failed to load multiple batteries: {str(e)}")
            raise DataLoaderError(f"Multi-battery loading failed: {str(e)}") from e

    def get_dataset_info(
        self, data: pd.DataFrame, name: str = "Dataset"
    ) -> DatasetInfo:
        """
        Generate comprehensive information about a dataset.

        Args:
            data: DataFrame to analyze
            name: Name of the dataset

        Returns:
            DatasetInfo object with comprehensive statistics
        """
        try:
            # Basic statistics
            n_samples, n_features = data.shape

            # Capacity statistics
            if "Capacity" in data.columns:
                capacity_range = (data["Capacity"].min(), data["Capacity"].max())
            else:
                capacity_range = (0.0, 0.0)

            # Cycle statistics
            if "id_cycle" in data.columns:
                cycle_range = (int(data["id_cycle"].min()), int(data["id_cycle"].max()))
            else:
                cycle_range = (0, 0)

            # Temperature statistics
            if "Temperature_measured" in data.columns:
                temp_range = (
                    data["Temperature_measured"].min(),
                    data["Temperature_measured"].max(),
                )
            else:
                temp_range = (0.0, 0.0)

            # Battery list
            if "Battery" in data.columns:
                batteries = sorted(data["Battery"].unique().tolist())
            else:
                batteries = ["Unknown"]

            return DatasetInfo(
                name=name,
                n_samples=n_samples,
                n_features=n_features,
                capacity_range=capacity_range,
                cycle_range=cycle_range,
                temperature_range=temp_range,
                batteries=batteries,
            )

        except Exception as e:
            logger.error(f"Failed to generate dataset info: {str(e)}")
            raise DataLoaderError(f"Dataset info generation failed: {str(e)}") from e

    def preprocess_for_modeling(
        self,
        data: pd.DataFrame,
        target_column: str = "Capacity",
        feature_columns: Optional[List[str]] = None,
        normalize: bool = True,
        add_derived_features: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess data for modeling applications.

        Args:
            data: Raw battery data
            target_column: Name of target column
            feature_columns: List of feature columns (auto-detected if None)
            normalize: Whether to normalize numerical features
            add_derived_features: Whether to add engineered features

        Returns:
            Tuple of (processed_data, feature_names)
        """
        try:
            logger.info("Preprocessing data for modeling")

            processed_data = data.copy()

            # Auto-detect feature columns if not provided
            if feature_columns is None:
                feature_columns = [
                    col
                    for col in processed_data.columns
                    if col != target_column
                    and processed_data[col].dtype in ["int64", "float64"]
                ]

            # Add derived features
            if add_derived_features:
                processed_data = self._add_derived_features(processed_data)

                # Update feature list with new features
                new_features = [
                    col
                    for col in processed_data.columns
                    if col not in data.columns and col != target_column
                ]
                feature_columns.extend(new_features)

            # Normalize features if requested
            if normalize:
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                processed_data[feature_columns] = scaler.fit_transform(
                    processed_data[feature_columns]
                )
                logger.debug("Applied StandardScaler normalization to features")

            logger.success(f"Preprocessing completed: {len(feature_columns)} features")
            return processed_data, feature_columns

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise DataLoaderError(f"Data preprocessing failed: {str(e)}") from e

    def _resolve_path(self, filepath: Union[str, Path]) -> Path:
        """Resolve file path relative to data directory."""
        filepath = Path(filepath)

        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        if not filepath.exists():
            raise DataLoaderError(f"File not found: {filepath}")

        return filepath

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and sanitize data."""
        logger.debug("Cleaning data")

        # Remove duplicates
        original_size = len(data)
        data = data.drop_duplicates()
        if len(data) < original_size:
            logger.debug(f"Removed {original_size - len(data)} duplicate rows")

        # Sanitize numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        data = sanitize_numeric_data(data, numeric_columns)

        # Sort by cycle and time if available
        if "id_cycle" in data.columns:
            sort_columns = ["id_cycle"]
            if "Time" in data.columns:
                sort_columns.append("Time")
            data = data.sort_values(sort_columns).reset_index(drop=True)

        return data

    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for better modeling."""
        logger.debug("Adding derived features")

        data = data.copy()

        # Cycle-based features
        if "id_cycle" in data.columns:
            data["cycle_normalized"] = data["id_cycle"] / data["id_cycle"].max()
            data["cycle_squared"] = data["id_cycle"] ** 2
            data["cycle_log"] = np.log1p(data["id_cycle"])

        # Temperature features
        if "Temperature_measured" in data.columns:
            data["temp_deviation"] = (
                data["Temperature_measured"] - data["Temperature_measured"].mean()
            )
            data["temp_squared"] = data["Temperature_measured"] ** 2

        # Time-based features
        if "Time" in data.columns:
            data["time_log"] = np.log1p(data["Time"])
            data["time_reciprocal"] = 1 / (data["Time"] + 1e-8)

        # Capacity features (if present)
        if "Capacity" in data.columns:
            # Rolling statistics (if enough data)
            if len(data) > 10:
                data["capacity_ma_5"] = (
                    data["Capacity"].rolling(window=5, min_periods=1).mean()
                )
                data["capacity_std_5"] = (
                    data["Capacity"].rolling(window=5, min_periods=1).std().fillna(0)
                )

        # Voltage features (if present)
        if "Voltage_measured" in data.columns:
            data["voltage_normalized"] = (
                data["Voltage_measured"] / data["Voltage_measured"].max()
            )

        # Current features (if present)
        if "Current_measured" in data.columns:
            data["current_abs"] = np.abs(data["Current_measured"])

        logger.debug(f"Added {len(data.columns) - len(data.columns)} derived features")
        return data

    def export_processed_data(
        self,
        data: pd.DataFrame,
        filepath: Union[str, Path],
        format: str = "csv",
        include_metadata: bool = True,
    ) -> None:
        """
        Export processed data to file.

        Args:
            data: Data to export
            filepath: Output file path
            format: Export format ("csv", "parquet", "hdf5")
            include_metadata: Whether to include dataset metadata
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Exporting data to {filepath} in {format} format")

            if format.lower() == "csv":
                data.to_csv(filepath, index=False)
            elif format.lower() == "parquet":
                data.to_parquet(filepath, index=False)
            elif format.lower() == "hdf5":
                data.to_hdf(filepath, key="battery_data", mode="w")
            else:
                raise DataLoaderError(f"Unsupported export format: {format}")

            # Export metadata if requested
            if include_metadata:
                dataset_info = self.get_dataset_info(data, filepath.stem)
                metadata_path = filepath.parent / f"{filepath.stem}_metadata.txt"

                with open(metadata_path, "w") as f:
                    f.write(str(dataset_info))
                    f.write(f"\n\nColumns:\n")
                    for col in data.columns:
                        f.write(f"  - {col}: {data[col].dtype}\n")

                logger.debug(f"Metadata exported to {metadata_path}")

            logger.success(f"Data export completed: {len(data):,} rows")

        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            raise DataLoaderError(f"Data export failed: {str(e)}") from e
