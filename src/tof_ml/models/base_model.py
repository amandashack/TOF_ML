#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Model class for the ML Provenance Tracker Framework.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseModelPlugin(ABC):
    """
    Base abstract class for all models.
    This replaces both ModelPlugin and the need for ModelFactory.
    """

    def __init__(self, **kwargs):
        """Initialize the model with hyperparameters."""
        self.hyperparameters = kwargs
        self.model = None
        self.is_fitted = False
        self.metadata = {
            "model_class": self.__class__.__name__,
            "hyperparameters": self.hyperparameters
        }

    @abstractmethod
    def fit(self, X, y, validation_data=None, **kwargs):
        """
        Train the model.

        Args:
            X: Feature data
            y: Target data
            validation_data: Optional validation data tuple
            **kwargs: Additional training parameters

        Returns:
            Training history or self
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the model.

        Args:
            X: Feature data

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def evaluate(self, X, y, **kwargs):
        """
        Evaluate the model.

        Args:
            X: Feature data
            y: Target data
            **kwargs: Additional evaluation parameters

        Returns:
            Evaluation metrics
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """
        Load a model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded model instance
        """
        pass

    def custom_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate custom evaluation metrics.
        Default implementation returns empty dict.

        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values

        Returns:
            Dictionary of custom metrics
        """
        return {}

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return self.hyperparameters.copy()

    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self.metadata.copy()


class KerasModelMixin:
    """Mixin for Keras/TensorFlow models."""

    def compile_model(self, optimizer, loss, metrics):
        """Compile a Keras model."""
        if hasattr(self, 'model') and self.model is not None:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def plot_model(self, to_file: str = "model.png"):
        """Plot model architecture."""
        if hasattr(self, 'model') and self.model is not None:
            from tensorflow.keras.utils import plot_model
            plot_model(self.model, to_file=to_file, show_shapes=True)

    def get_model_summary(self) -> str:
        """Get model summary as string."""
        if hasattr(self, 'model') and self.model is not None:
            import io
            stream = io.StringIO()
            self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
            return stream.getvalue()
        return "Model not built yet"


class SklearnModelMixin:
    """Mixin for scikit-learn models."""

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if hasattr(self, 'model') and hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if hasattr(self, 'model') and hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}