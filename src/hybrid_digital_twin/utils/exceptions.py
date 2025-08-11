"""
Custom exceptions for the hybrid digital twin framework.

This module defines specific exception types used throughout the framework
to provide clear error handling and debugging information.
"""


class DigitalTwinError(Exception):
    """Base exception for all digital twin related errors."""
    pass


class ModelError(DigitalTwinError):
    """Exception raised for model-related errors."""
    pass


class ModelNotTrainedError(ModelError):
    """Exception raised when attempting to use an untrained model."""
    pass


class InvalidDataError(DigitalTwinError):
    """Exception raised for invalid or malformed input data."""
    pass


class InvalidParameterError(DigitalTwinError):
    """Exception raised for invalid parameter values."""
    pass


class ConfigurationError(DigitalTwinError):
    """Exception raised for configuration-related errors."""
    pass


class DataLoaderError(DigitalTwinError):
    """Exception raised for data loading and processing errors."""
    pass


class VisualizationError(DigitalTwinError):
    """Exception raised for visualization-related errors."""
    pass