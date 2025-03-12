# src/tof_ml/pipeline/tof_dataset.py
import os
import numpy as np
import h5py
import json
import logging
import yaml
from typing import Dict, List, Tuple, Optional, Union, Callable, Iterator, Any
from datetime import datetime
import hashlib
import glob
import multiprocessing as mp
from queue import Queue
from threading import Thread
import tensorflow as tf

from src.tof_ml.transforms.transform_pipeline import TransformPipeline, TRANSFORM_REGISTRY
from src.tof_ml.transforms.common_transforms import *

logger = logging.getLogger("tof_dataset")


class TOFDataset:
    """
    Dataset class for TOF data, inspired by TensorFlow's Dataset API.
    Provides iterable access to TOF data with built-in transformations.
    """

    def __init__(self,
                 h5_file_path: Optional[str] = None,
                 h5_group_path: str = 'data',
                 batch_size: int = 32,
                 shuffle_buffer: int = 1000,
                 prefetch_buffer: int = 5,
                 feature_columns: Optional[List[str]] = None,
                 label_columns: Optional[List[str]] = None,
                 pipeline: Optional[TransformPipeline] = None):
        """
        Initialize the TOF dataset.

        Args:
            h5_file_path: Path to the H5 file containing processed data
            h5_group_path: Path to the group within the H5 file
            batch_size: Number of samples per batch
            shuffle_buffer: Size of the shuffle buffer
            prefetch_buffer: Number of batches to prefetch
            feature_columns: Names or indices of columns to use as features
            label_columns: Names or indices of columns to use as labels
            pipeline: Optional transformation pipeline to apply
        """
        self.h5_file_path = h5_file_path
        self.h5_group_path = h5_group_path
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.pipeline = pipeline

        # These will be initialized when loading data
        self.data = None
        self.features = None
        self.labels = None
        self.column_names = []
        self.column_indices = []
        self._column_mapping = {}
        self.num_samples = 0
        self.tf_dataset = None

        # Load the data if path is provided
        if h5_file_path:
            self.load_data()

    def load_data(self) -> 'TOFDataset':
        """
        Load data from the H5 file.

        Returns:
            Self for chaining
        """
        if not self.h5_file_path or not os.path.exists(self.h5_file_path):
            raise ValueError(f"Invalid or missing H5 file path: {self.h5_file_path}")

        logger.info(f"Loading data from {self.h5_file_path}, group {self.h5_group_path}")

        with h5py.File(self.h5_file_path, 'r') as f:
            # Get the appropriate group
            if self.h5_group_path in f:
                group = f[self.h5_group_path]
            else:
                # If group not found, try to find a dataset with that name
                if self.h5_group_path in f and isinstance(f[self.h5_group_path], h5py.Dataset):
                    group = f
                else:
                    raise ValueError(f"Group or dataset {self.h5_group_path} not found in file")

            # Load column mapping from attributes
            if 'column_info' in f.attrs:
                column_info = json.loads(f.attrs['column_info'])
                self.column_names = column_info.get('names', [])
                self.column_indices = column_info.get('indices', [])

                # Create mapping from name to index
                self._column_mapping = {name: idx for name, idx in zip(self.column_names, self.column_indices)}

            # Load the dataset
            if isinstance(group, h5py.Group) and 'data' in group:
                self.data = group['data'][:]
            elif isinstance(group, h5py.Dataset):
                self.data = group[:]
            else:
                raise ValueError(f"No dataset found at {self.h5_group_path}")

            self.num_samples = self.data.shape[0]
            logger.info(f"Loaded {self.num_samples} samples with shape {self.data.shape}")

            # Load pipeline if it exists in file metadata
            if self.pipeline is None and 'pipeline_metadata' in f.attrs:
                try:
                    pipeline_data = json.loads(f.attrs['pipeline_metadata'])
                    self.pipeline = TransformPipeline.deserialize(pipeline_data, TRANSFORM_REGISTRY)
                    logger.info(f"Loaded pipeline from file: {self.pipeline.name}")
                except Exception as e:
                    logger.warning(f"Failed to load pipeline from file: {e}")

        # Split data into features and labels
        self._split_features_labels()

        # Apply any additional transformations if pipeline is provided
        if self.pipeline:
            self._apply_pipeline()

        return self

    def _split_features_labels(self):
        """Split data into features and labels based on column specifications"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Convert column names to indices if provided
        feature_indices = self._resolve_column_indices(self.feature_columns)
        label_indices = self._resolve_column_indices(self.label_columns)

        # Default behavior: if no columns specified, use all for features, none for labels
        if not feature_indices and not label_indices:
            self.features = self.data
            self.labels = None
            return

        # Create feature and label arrays
        if feature_indices:
            self.features = self.data[:, feature_indices]
        else:
            self.features = None

        if label_indices:
            self.labels = self.data[:, label_indices]
        else:
            self.labels = None

    def _resolve_column_indices(self, columns) -> List[int]:
        """
        Resolve column specifications to indices.

        Args:
            columns: List of column names or indices

        Returns:
            List of column indices
        """
        if not columns:
            return []

        indices = []
        for col in columns:
            if isinstance(col, int):
                indices.append(col)
            elif isinstance(col, str) and col in self._column_mapping:
                indices.append(self._column_mapping[col])
            else:
                logger.warning(f"Column not found: {col}")

        return indices

    def _apply_pipeline(self):
        """Apply the transformation pipeline to the data"""
        if self.pipeline and self.features is not None:
            try:
                self.features = self.pipeline.transform(self.features)
                logger.info(f"Applied pipeline transformations. New feature shape: {self.features.shape}")
            except Exception as e:
                logger.error(f"Error applying pipeline: {e}")

    def create_tf_dataset(self) -> tf.data.Dataset:
        """
        Create a TensorFlow Dataset from the loaded data.

        Returns:
            TensorFlow Dataset object
        """
        if self.features is None:
            raise ValueError("Features not available. Load data and split features/labels first.")

        # Convert numpy arrays to TensorFlow tensors
        if self.labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(self.features)

        # Apply dataset transformations
        if self.shuffle_buffer > 0:
            dataset = dataset.shuffle(buffer_size=min(self.shuffle_buffer, self.num_samples))

        dataset = dataset.batch(self.batch_size)

        if self.prefetch_buffer > 0:
            dataset = dataset.prefetch(self.prefetch_buffer)

        self.tf_dataset = dataset
        return dataset

    def batch_iterator(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Create a simple batch iterator for the dataset.

        Returns:
            Iterator yielding batches of (features, labels)
        """
        if self.features is None:
            raise ValueError("Features not available. Load data and split features/labels first.")

        # If shuffle enabled, create a shuffled index array
        indices = np.arange(self.num_samples)
        if self.shuffle_buffer > 0:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]

            features_batch = self.features[batch_indices]

            if self.labels is not None:
                labels_batch = self.labels[batch_indices]
                yield features_batch, labels_batch
            else:
                yield features_batch, None

    def with_batch_size(self, batch_size: int) -> 'TOFDataset':
        """
        Create a new dataset with the specified batch size.

        Args:
            batch_size: New batch size

        Returns:
            New TOFDataset instance
        """
        new_dataset = TOFDataset(
            h5_file_path=self.h5_file_path,
            h5_group_path=self.h5_group_path,
            batch_size=batch_size,
            shuffle_buffer=self.shuffle_buffer,
            prefetch_buffer=self.prefetch_buffer,
            feature_columns=self.feature_columns,
            label_columns=self.label_columns,
            pipeline=self.pipeline
        )

        # Copy already loaded data
        new_dataset.data = self.data
        new_dataset.features = self.features
        new_dataset.labels = self.labels
        new_dataset.column_names = self.column_names
        new_dataset.column_indices = self.column_indices
        new_dataset._column_mapping = self._column_mapping
        new_dataset.num_samples = self.num_samples

        return new_dataset

    def with_shuffle(self, buffer_size: int) -> 'TOFDataset':
        """
        Create a new dataset with the specified shuffle buffer size.

        Args:
            buffer_size: Size of the shuffle buffer

        Returns:
            New TOFDataset instance
        """
        new_dataset = self.with_batch_size(self.batch_size)
        new_dataset.shuffle_buffer = buffer_size
        return new_dataset

    def with_prefetch(self, buffer_size: int) -> 'TOFDataset':
        """
        Create a new dataset with the specified prefetch buffer size.

        Args:
            buffer_size: Size of the prefetch buffer

        Returns:
            New TOFDataset instance
        """
        new_dataset = self.with_batch_size(self.batch_size)
        new_dataset.prefetch_buffer = buffer_size
        return new_dataset

    def with_pipeline(self, pipeline: TransformPipeline) -> 'TOFDataset':
        """
        Create a new dataset with the specified transformation pipeline.

        Args:
            pipeline: Transformation pipeline to apply

        Returns:
            New TOFDataset instance
        """
        new_dataset = self.with_batch_size(self.batch_size)
        new_dataset.pipeline = pipeline

        # Apply the pipeline to the features
        if new_dataset.features is not None:
            new_dataset._apply_pipeline()

        return new_dataset

    def with_columns(self, feature_columns: List[Union[str, int]],
                     label_columns: Optional[List[Union[str, int]]] = None) -> 'TOFDataset':
        """
        Create a new dataset with the specified feature and label columns.

        Args:
            feature_columns: Column names or indices for features
            label_columns: Column names or indices for labels

        Returns:
            New TOFDataset instance
        """
        new_dataset = self.with_batch_size(self.batch_size)
        new_dataset.feature_columns = feature_columns
        new_dataset.label_columns = label_columns

        # Re-split features and labels
        if new_dataset.data is not None:
            new_dataset._split_features_labels()

            # Re-apply pipeline if it exists
            if new_dataset.pipeline:
                new_dataset._apply_pipeline()

        return new_dataset

    def take(self, n: int) -> 'TOFDataset':
        """
        Create a new dataset with only the first n samples.

        Args:
            n: Number of samples to take

        Returns:
            New TOFDataset instance
        """
        new_dataset = self.with_batch_size(self.batch_size)

        if self.data is not None:
            n = min(n, self.num_samples)
            new_dataset.data = self.data[:n]
            new_dataset.num_samples = n

            if self.features is not None:
                new_dataset.features = self.features[:n]

            if self.labels is not None:
                new_dataset.labels = self.labels[:n]

        return new_dataset

    def head(self, n: int = 5) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Return the first n samples.

        Args:
            n: Number of samples to return

        Returns:
            Tuple of (features, labels) arrays
        """
        if self.features is None:
            raise ValueError("Features not available. Load data first.")

        n = min(n, self.num_samples)

        if self.labels is not None:
            return self.features[:n], self.labels[:n]
        else:
            return self.features[:n], None

    def sample(self, n: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Return n random samples.

        Args:
            n: Number of samples to return

        Returns:
            Tuple of (features, labels) arrays
        """
        if self.features is None:
            raise ValueError("Features not available. Load data first.")

        n = min(n, self.num_samples)
        indices = np.random.choice(self.num_samples, n, replace=False)

        if self.labels is not None:
            return self.features[indices], self.labels[indices]
        else:
            return self.features[indices], None

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return self.num_samples

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Make the dataset iterable, returning batches"""
        return self.batch_iterator()

    @classmethod
    def load_from_processed_file(cls, file_path: str,
                                 group_path: str = 'data',
                                 batch_size: int = 32,
                                 feature_columns: Optional[List[str]] = None,
                                 label_columns: Optional[List[str]] = None) -> 'TOFDataset':
        """
        Create a dataset from a processed H5 file.

        Args:
            file_path: Path to the H5 file
            group_path: Path to the group within the H5 file
            batch_size: Batch size
            feature_columns: Column names or indices for features
            label_columns: Column names or indices for labels

        Returns:
            New TOFDataset instance
        """
        return cls(
            h5_file_path=file_path,
            h5_group_path=group_path,
            batch_size=batch_size,
            feature_columns=feature_columns,
            label_columns=label_columns
        )