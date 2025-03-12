# src/tof_ml/data/enhanced_data_loader.py
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import h5py
import os
import json
from datetime import datetime

from src.tof_ml.data.base_data import BaseDataLoader
from src.tof_ml.transforms.transform_pipeline import TransformPipeline, TRANSFORM_REGISTRY
from src.tof_ml.transforms.base_transform import BaseTransform


class EnhancedDataLoader(BaseDataLoader):
    """
    An enhanced data loader that supports transformation pipelines
    """

    def __init__(self, config: dict):
        """
        Initialize the data loader with a configuration dictionary.
        """
        super().__init__(config)
        self.pipeline = None
        self.raw_data = None
        self.transformed_data = None
        self.dataset_metadata = {}

    def set_pipeline(self, pipeline: TransformPipeline) -> 'EnhancedDataLoader':
        """Set the transformation pipeline"""
        self.pipeline = pipeline
        return self

    def load_data(self) -> np.ndarray:
        """
        Load the raw data from the source.
        This should be overridden by subclasses.
        """
        self.raw_data = self._load_raw_data()

        # Apply transforms if available
        if self.pipeline is not None and self.raw_data is not None:
            self.transformed_data = self.pipeline.transform(self.raw_data)
            return self.transformed_data

        return self.raw_data

    @abstractmethod
    def _load_raw_data(self) -> np.ndarray:
        """
        Load the raw data from the source.
        This should be implemented by subclasses.
        """
        pass

    def save_dataset(self, file_path: str, name: str = None) -> str:
        """
        Save the dataset (both raw and transformed) to an H5 file

        Args:
            file_path: Path to the H5 file
            name: Optional name for the dataset group, defaults to class name

        Returns:
            The group name where the data was stored
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        name = name or f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with h5py.File(file_path, 'a') as f:
            # Create group for this dataset if it doesn't exist
            if name in f:
                del f[name]

            group = f.create_group(name)

            # Store raw data
            raw_group = group.create_group('raw_data')
            raw_dataset = raw_group.create_dataset('data', data=self.raw_data)
            raw_dataset.attrs['shape'] = self.raw_data.shape
            raw_dataset.attrs['columns'] = json.dumps(
                {v: k for k, v in self.column_mapping.items()}
            )

            # Store dataset metadata
            for key, value in self.dataset_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    group.attrs[key] = value
                else:
                    group.attrs[key] = json.dumps(value)

            # Store class name and config
            group.attrs['loader_class'] = self.__class__.__name__
            group.attrs['config'] = json.dumps(self.config)

            # Store transformed data if available
            if self.transformed_data is not None and self.pipeline is not None:
                # Save the pipeline
                pipeline_group = group.create_group('pipeline')
                pipeline_metadata = json.dumps(self.pipeline.serialize())
                pipeline_group.attrs['metadata'] = pipeline_metadata

                # Save the transformed data
                transformed_group = group.create_group('transformed_data')
                transformed_dataset = transformed_group.create_dataset('data', data=self.transformed_data)
                transformed_dataset.attrs['shape'] = self.transformed_data.shape

        return name

    @classmethod
    def load_dataset(cls, file_path: str, group_name: str) -> Tuple['EnhancedDataLoader', np.ndarray]:
        """
        Load a dataset from an H5 file

        Args:
            file_path: Path to the H5 file
            group_name: Name of the dataset group

        Returns:
            A tuple of (loader, transformed_data)
        """
        with h5py.File(file_path, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group {group_name} not found in file {file_path}")

            group = f[group_name]

            # Get loader class and config
            loader_class_name = group.attrs['loader_class']
            config = json.loads(group.attrs['config'])

            # Get dataset metadata
            metadata = {}
            for key, value in group.attrs.items():
                if key not in ['loader_class', 'config']:
                    try:
                        metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        metadata[key] = value

            # Import the loader class dynamically
            import importlib
            module_path = f"src.tof_ml.data.{loader_class_name.lower()}"
            try:
                module = importlib.import_module(module_path)
                LoaderClass = getattr(module, loader_class_name)
            except (ImportError, AttributeError):
                # Fallback to the current class if specific loader not found
                LoaderClass = cls

            # Instantiate the loader
            loader = LoaderClass(config=config)
            loader.dataset_metadata = metadata

            # Load raw data
            if 'raw_data' in group and 'data' in group['raw_data']:
                loader.raw_data = group['raw_data']['data'][:]

            # Load pipeline if available
            if 'pipeline' in group:
                pipeline_metadata = json.loads(group['pipeline'].attrs['metadata'])
                loader.pipeline = TransformPipeline.deserialize(pipeline_metadata, TRANSFORM_REGISTRY)

            # Load transformed data if available
            if 'transformed_data' in group and 'data' in group['transformed_data']:
                loader.transformed_data = group['transformed_data']['data'][:]
            elif loader.raw_data is not None and loader.pipeline is not None:
                # If transformed data not stored but we have raw data and pipeline,
                # we can regenerate the transformed data
                loader.transformed_data = loader.pipeline.transform(loader.raw_data)

            return loader, loader.transformed_data or loader.raw_data