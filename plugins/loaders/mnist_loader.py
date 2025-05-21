#!/usr/bin/env python3
"""MNIST Data Loader Plugin for the ML Provenance Tracker Framework."""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf

from src.tof_ml.data.base_data_loader import BaseDataLoader
from src.tof_ml.pipeline.plugins.interfaces import DataLoaderPlugin

logger = logging.getLogger(__name__)


class MNISTConvLoader(BaseDataLoader, DataLoaderPlugin):
    """Data loader plugin for MNIST dataset for CNN using TensorFlow."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.flatten = False  # We don't want to flatten for CNN
        self.normalize = kwargs.get('normalize', True)
        self.one_hot = kwargs.get('one_hot', False)
        logger.info("MNISTConvLoader initialized")

    def load_data(self) -> np.ndarray:
        """Load MNIST data from TensorFlow datasets."""
        logger.info("Loading MNIST dataset for CNN")

        # Load from TensorFlow
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Combine train and test data
        x_data = np.concatenate([x_train, x_test], axis=0)
        y_data = np.concatenate([y_train, y_test], axis=0)

        # Process features - reshape to include channel dimension
        x_data = x_data.reshape(x_data.shape[0], 28, 28, 1)

        if self.normalize:
            x_data = x_data.astype('float32') / 255.0

        # Process targets
        if self.one_hot:
            y_data = tf.keras.utils.to_categorical(y_data, 10)
        else:
            y_data = y_data.reshape(-1, 1)

        # Store processed data for later feature extraction
        self._features_cache = x_data
        self._targets_cache = y_data
        self._data_loaded = True

        # Update metadata
        self.metadata.update({
            "dataset_type": "MNIST",
            "sample_count": len(x_data),
            "feature_shape": (28, 28, 1),
            "target_count": 10 if self.one_hot else 1
        })

        # Create a special "combined" representation for the data manager
        # that will be unpacked correctly by extract_features_and_targets
        combined_data = {
            "_x_data": x_data,
            "_y_data": y_data,
            "_is_cnn_data": True
        }

        logger.info(f"MNIST data loaded for CNN with features shape: {x_data.shape} and targets shape: {y_data.shape}")
        return combined_data

    def extract_features_and_targets(self, data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from MNIST data."""
        if data is None:
            if self._data_loaded and hasattr(self, '_features_cache') and hasattr(self, '_targets_cache'):
                return self._features_cache, self._targets_cache
            data = self.load_data()

        # Check if this is our special combined format
        if isinstance(data, dict) and "_is_cnn_data" in data:
            return data["_x_data"], data["_y_data"]

        # Fallback for other formats
        logger.warning("Data format not recognized in extract_features_and_targets")
        if self._data_loaded and hasattr(self, '_features_cache') and hasattr(self, '_targets_cache'):
            return self._features_cache, self._targets_cache

        # Last resort: reload data
        return self.load_data()["_x_data"], self.load_data()["_y_data"]

class MNISTLoader(BaseDataLoader, DataLoaderPlugin):
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