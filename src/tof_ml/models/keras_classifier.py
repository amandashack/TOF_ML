#!/usr/bin/env python3
"""Keras classifier model for the ML Provenance Tracker Framework."""

import logging
import tensorflow as tf
from typing import Dict, Any, Optional, List, Union
import numpy as np

logger = logging.getLogger(__name__)


class MLPKerasClassifier:
    """Multi-layer perceptron classifier using Keras."""

    def __init__(self,
                 hidden_layers: List[int] = [128, 64],
                 activations: List[str] = ["relu", "relu"],
                 output_activation: str = "softmax",
                 output_units: int = 10,
                 learning_rate: float = 0.001,
                 optimizer_name: str = "Adam",
                 loss: str = "sparse_categorical_crossentropy",
                 metrics: List[str] = ["accuracy"],
                 **kwargs):
        """Initialize the model."""
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.output_activation = output_activation
        self.output_units = output_units
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.loss = loss
        self.metrics = metrics

        # Define metrics names for access in evaluate method
        self.metrics_names = ['loss'] + metrics

        # Build the model
        self.model = self._build_model()
        logger.info("MLPKerasClassifier initialized")

    def _build_model(self) -> tf.keras.Model:
        """Build the Keras model."""
        model = tf.keras.Sequential()

        # Add hidden layers
        for i, (units, activation) in enumerate(zip(self.hidden_layers, self.activations)):
            if i == 0:
                model.add(tf.keras.layers.Dense(units, activation=activation, input_shape=(784,)))
            else:
                model.add(tf.keras.layers.Dense(units, activation=activation))

        # Add output layer
        model.add(tf.keras.layers.Dense(self.output_units, activation=self.output_activation))

        # Compile model
        optimizer = tf.keras.optimizers.get(self.optimizer_name)
        if hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate = self.learning_rate

        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)

        return model

    def fit(self, X, y, **kwargs):
        """Train the model."""
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def evaluate(self, X, y, **kwargs):
        """Evaluate the model.

        Returns:
            Either a scalar test loss (if no metrics) or
            a list of [loss, metric1, metric2, ...] values
        """
        result = self.model.evaluate(X, y, **kwargs)

        # Ensure metrics_names matches the actual metrics
        # TF 2.x uses different approaches in different versions
        if hasattr(self.model, 'metrics_names'):
            self.metrics_names = self.model.metrics_names

        return result

    def save(self, path):
        """Save the model."""
        self.model.save(path)

    def custom_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate classification-specific metrics.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            Dictionary of classification metrics
        """
        metrics = {}

        # Convert to class indices if needed
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = y_pred

        # Reshape y_true if needed
        if len(y_true.shape) > 1 and y_true.shape[1] == 1:
            y_true = y_true.ravel()

        # Calculate accuracy
        accuracy = np.mean(y_pred_classes == y_true)
        metrics["accuracy"] = float(accuracy)

        try:
            # Try to use sklearn metrics if available
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

            # For multiclass classification, use 'macro' average by default
            precision = precision_score(y_true, y_pred_classes, average='macro')
            recall = recall_score(y_true, y_pred_classes, average='macro')
            f1 = f1_score(y_true, y_pred_classes, average='macro')

            metrics["precision"] = float(precision)
            metrics["recall"] = float(recall)
            metrics["f1"] = float(f1)

            # Add confusion matrix (flattened)
            cm = confusion_matrix(y_true, y_pred_classes)
            metrics["confusion_matrix"] = cm.tolist()

        except ImportError:
            # If sklearn is not available, compute accuracy only
            logger.warning("sklearn not available, calculating only basic metrics")

        return metrics

    def get_loss_value(self, metrics: Dict[str, float]) -> float:
        """
        Get the primary loss value from metrics.
        For classification, negative accuracy serves as loss.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Loss value (lower is better)
        """
        # For classification, use negative accuracy as loss
        if "accuracy" in metrics:
            return -metrics["accuracy"]
        # Fallback to mse if available
        elif "mse" in metrics:
            return metrics["mse"]
        # Generic fallback
        else:
            return 0.0

    @classmethod
    def load(cls, path):
        """Load a saved model."""
        instance = cls()
        instance.model = tf.keras.models.load_model(path)

        # Update metrics_names from loaded model
        if hasattr(instance.model, 'metrics_names'):
            instance.metrics_names = instance.model.metrics_names

        return instance

    # This is the key addition to handle the metrics_names attribute
    @property
    def metrics_names(self):
        """Get the metrics names from the underlying Keras model if available."""
        if hasattr(self.model, 'metrics_names'):
            return self.model.metrics_names
        return self._metrics_names

    @metrics_names.setter
    def metrics_names(self, value):
        """Set metrics names."""
        self._metrics_names = value