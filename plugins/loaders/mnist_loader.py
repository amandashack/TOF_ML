# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST Data Loader Plugin for the ML Provenance Tracker Framework.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
from src.tof_ml.data.base_data_loader import BaseDataLoaderPlugin
import logging

logger = logging.getLogger(__name__)


class MNISTConvLoader(BaseDataLoaderPlugin):
    """
    Data loader for MNIST dataset optimized for convolutional neural networks.
    Inherits from DataLoaderPlugin which inherits from BaseDataLoader.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initialize the MNIST loader.

        Args:
            config: Configuration dictionary
            **kwargs: Additional parameters
        """
        # Call parent __init__ first
        super().__init__(config, **kwargs)

        # MNIST-specific parameters
        self.normalize = kwargs.get("normalize", True)
        self.one_hot = kwargs.get("one_hot", False)
        self.flatten = kwargs.get("flatten", False)

        # Update metadata with MNIST-specific info
        self.metadata.update({
            "normalize": self.normalize,
            "one_hot": self.one_hot,
            "flatten": self.flatten,
            "dataset_type": "MNIST"
        })

    def _validate_config(self) -> None:
        """
        Validate MNIST-specific configuration.
        """
        # For MNIST, we don't need a dataset path since it's downloaded
        # Override the parent validation
        pass

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MNIST data from TensorFlow datasets.

        Returns:
            Tuple of (features, targets) combined from train and test sets
        """
        # Load MNIST data
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Combine train and test for splitting by the framework
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        # Add channel dimension for CNN (samples, 28, 28) -> (samples, 28, 28, 1)
        if not self.flatten:
            X = np.expand_dims(X, axis=-1)
        else:
            # Flatten for non-CNN models
            X = X.reshape(X.shape[0], -1)

        # Normalize pixel values to [0, 1]
        if self.normalize:
            X = X.astype('float32') / 255.0

        # One-hot encode labels if requested
        if self.one_hot:
            y = tf.keras.utils.to_categorical(y, num_classes=10)

        # Store in cache (from BaseDataLoader)
        self._data_cache = (X, y)
        self._data_loaded = True

        # Update metadata
        self.metadata.update({
            "total_samples": X.shape[0],
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "input_shape": X.shape[1:],
            "num_classes": 10,
            "data_type": "image",
            "image_shape": (28, 28, 1) if not self.flatten else (784,)
        })

        return (X, y)

    def extract_features_and_targets(self, data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Extract feature and target arrays from data.

        For MNISTConvLoader, the data is already in (X, y) tuple format,
        so we just need to return it as-is.

        Args:
            data: Optional data tuple. If None, uses loaded data.

        Returns:
            Tuple of (features, targets)
        """
        if data is None:
            if self._data_cache is None:
                self._data_cache = self.load_data()
                self._data_loaded = True
            data = self._data_cache

        # Handle tuple data (already split) - this is the expected case for MNISTConvLoader
        if isinstance(data, tuple) and len(data) == 2:
            X, y = data
            return X, y

        # Handle array data (needs splitting) - fallback case
        if isinstance(data, np.ndarray):
            # This shouldn't happen with MNISTConvLoader, but provide fallback
            if len(data.shape) == 2:  # Flattened format
                # Assume all columns except last are features
                features = data[:, :-1]
                targets = data[:, -1]
                return features, targets
            else:
                raise ValueError(f"Unexpected data format for CNN loader: {data.shape}")

        raise ValueError(f"Data must be ndarray or tuple, got {type(data)}")


class MNISTLoader(BaseDataLoaderPlugin):
    """Data loader plugin for MNIST dataset using TensorFlow."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.flatten = kwargs.get('flatten', True)
        self.normalize = kwargs.get('normalize', True)
        self.one_hot = kwargs.get('one_hot', False)
        logger.info("MNISTLoader initialized")

    def load_data(self) -> np.ndarray:
        """Load MNIST data from TensorFlow datasets."""
        logger.info("Loading MNIST dataset")

        # Load from TensorFlow
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Combine train and test data
        x_data = np.concatenate([x_train, x_test], axis=0)
        y_data = np.concatenate([y_train, y_test], axis=0)

        # Process features
        if self.flatten:
            x_data = x_data.reshape(x_data.shape[0], -1)

        if self.normalize:
            x_data = x_data.astype('float32') / 255.0

        # Process targets
        if self.one_hot:
            y_data = tf.keras.utils.to_categorical(y_data, 10)
        else:
            y_data = y_data.reshape(-1, 1)

        # Combine features and targets
        data = np.hstack([x_data, y_data])

        # Apply sampling if needed
        if self.n_samples is not None and self.n_samples < len(data):
            indices = np.random.choice(len(data), self.n_samples, replace=False)
            data = data[indices]

        self._data_cache = data
        self._data_loaded = True

        # Update metadata
        self.metadata.update({
            "dataset_type": "MNIST",
            "sample_count": len(data),
            "feature_count": 784 if self.flatten else (28, 28),
            "target_count": 10 if self.one_hot else 1
        })

        logger.info(f"MNIST data loaded with shape: {data.shape}")
        return data

    def extract_features_and_targets(self, data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from MNIST data."""
        if data is None:
            if self._data_cache is None:
                self.load_data()
            data = self._data_cache

        if self.one_hot:
            features = data[:, :-10]
            targets = data[:, -10:]
        else:
            features = data[:, :-1]
            targets = data[:, -1].astype(int)

        return features, targets