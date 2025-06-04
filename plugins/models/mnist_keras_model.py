#!/usr/bin/env python3
"""MNIST model plugin for the ML Provenance Tracker Framework."""

import logging
import tensorflow as tf
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import keras

from src.tof_ml.models.base_model import BaseModelPlugin, KerasModelMixin

logger = logging.getLogger(__name__)


class MNISTConvModel(BaseModelPlugin, KerasModelMixin):
    """CNN model for MNIST using the new structure."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.history = None

        # Model architecture parameters
        self.conv_filters = kwargs.get("conv_filters", [32, 64])
        self.kernel_sizes = kwargs.get("kernel_sizes", [3, 3])
        self.pool_sizes = kwargs.get("pool_sizes", [2, 2])
        self.dense_layers = kwargs.get("dense_layers", [128])
        self.dropout_rate = kwargs.get("dropout_rate", 0.3)
        self.activations = kwargs.get("activations", ["relu", "relu", "relu"])
        self.output_activation = kwargs.get("output_activation", "softmax")
        self.output_units = kwargs.get("output_units", 10)

        # Training parameters
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.optimizer_name = kwargs.get("optimizer_name", "Adam")
        self.loss = kwargs.get("loss", "sparse_categorical_crossentropy")
        self.metrics = kwargs.get("metrics", ["accuracy"])

        # Model instance
        self.model = None
        self.input_shape = None

        # Store hyperparameters for tracking
        self.hyperparameters = {
            "conv_filters": self.conv_filters,
            "kernel_sizes": self.kernel_sizes,
            "pool_sizes": self.pool_sizes,
            "dense_layers": self.dense_layers,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer_name
        }

    def _build_model(self, input_shape: Tuple[int, ...]):
        """Build the CNN architecture."""
        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Input(shape=input_shape))

        # Convolutional layers
        for i, (filters, kernel_size, pool_size) in enumerate(
                zip(self.conv_filters, self.kernel_sizes, self.pool_sizes)
        ):
            # Conv layer
            model.add(tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation=self.activations[i] if i < len(self.activations) else 'relu',
                padding='same'
            ))

            # Pooling layer
            if pool_size > 1:
                model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

            # Optional batch normalization
            model.add(tf.keras.layers.BatchNormalization())

        # Flatten for dense layers
        model.add(tf.keras.layers.Flatten())

        # Dense layers
        for i, units in enumerate(self.dense_layers):
            model.add(tf.keras.layers.Dense(
                units,
                activation=self.activations[len(self.conv_filters) + i]
                if len(self.conv_filters) + i < len(self.activations) else 'relu'
            ))

            # Dropout
            if self.dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(self.dropout_rate))

        # Output layer
        model.add(tf.keras.layers.Dense(
            self.output_units,
            activation=self.output_activation
        ))

        # Compile the model
        optimizer = tf.keras.optimizers.get(self.optimizer_name)
        if hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate = self.learning_rate

        model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=self.metrics
        )

        # FIXED: Assign the built model to self.model
        self.model = model
        return model

    def fit(self, X, y, validation_data=None, **kwargs):
        """Train the model."""
        if self.model is None:
            self._build_model(X.shape[1:])

        # Handle parameter conflicts - trainer passes these as kwargs
        fit_kwargs = kwargs.copy()
        epochs = fit_kwargs.pop('epochs', 10)
        batch_size = fit_kwargs.pop('batch_size', 32)

        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            **fit_kwargs
        )
        self.is_fitted = True
        return self.history

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def evaluate(self, X, y, **kwargs):
        """Evaluate the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        return self.model.evaluate(X, y, **kwargs)

    def save(self, path: str):
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)

        # Save hyperparameters
        import json
        with open(f"{path}_hyperparameters.json", 'w') as f:
            json.dump(self.hyperparameters, f)

    @classmethod
    def load(cls, path: str):
        """Load a saved model."""
        model = keras.models.load_model(path)

        # Load hyperparameters if available
        import json
        import os
        hyperparam_path = f"{path}_hyperparameters.json"
        if os.path.exists(hyperparam_path):
            with open(hyperparam_path, 'r') as f:
                hyperparameters = json.load(f)
        else:
            hyperparameters = {}

        # Create instance
        instance = cls(**hyperparameters)
        instance.model = model
        instance.is_fitted = True

        return instance


class MNISTKerasModel(BaseModelPlugin):
    """Neural network model for MNIST classification using Keras."""

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
        logger.info("MNISTKerasModel initialized")

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
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path)

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