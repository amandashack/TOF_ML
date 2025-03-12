# src/tof_ml/transforms/base_transform.py
from abc import ABC, abstractmethod
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple


class BaseTransform(ABC):
    """Base class for all data transformations"""

    def __init__(self, name: str = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.params = kwargs
        self._is_fitted = False
        self._metadata = {}  # For storing fit statistics or other metadata

    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> 'BaseTransform':
        """Fit the transform parameters to the data"""
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the transformation to the data"""
        pass

    def fit_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Fit and then transform the data"""
        self.fit(data, **kwargs)
        return self.transform(data)

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of this transform"""
        return self.params.copy()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the fitting process"""
        return self._metadata.copy()

    def serialize(self) -> Dict[str, Any]:
        """Serialize the transform to a dictionary"""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "params": self.params,
            "metadata": self._metadata,
            "is_fitted": self._is_fitted
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any], registry: Dict[str, type]) -> 'BaseTransform':
        """Deserialize a transform from a dictionary"""
        transform_type = data["type"]
        transform_class = registry.get(transform_type)
        if transform_class is None:
            raise ValueError(f"Unknown transform type: {transform_type}")

        instance = transform_class(name=data["name"], **data["params"])
        instance._metadata = data["metadata"]
        instance._is_fitted = data["is_fitted"]
        return instance


# src/tof_ml/transforms/transform_pipeline.py
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import json
import h5py
import os
import uuid
from datetime import datetime

from src.tof_ml.transforms.base_transform import BaseTransform
from src.tof_ml.data.column_mapping import COLUMN_MAPPING


class TransformPipeline:
    """A pipeline of transformations that can be applied sequentially"""

    def __init__(self, name: str = None):
        self.name = name or f"pipeline_{uuid.uuid4().hex[:8]}"
        self.transforms: List[BaseTransform] = []
        self.history: List[Dict[str, Any]] = []
        self.column_mapping = COLUMN_MAPPING.copy()
        self.creation_timestamp = datetime.now().isoformat()
        self.last_modified_timestamp = self.creation_timestamp

    def add_transform(self, transform: BaseTransform) -> 'TransformPipeline':
        """Add a transform to the pipeline"""
        self.transforms.append(transform)
        self.last_modified_timestamp = datetime.now().isoformat()
        return self

    def fit(self, data: np.ndarray, **kwargs) -> 'TransformPipeline':
        """Fit all transforms in the pipeline"""
        transformed_data = data.copy()
        for transform in self.transforms:
            transform.fit(transformed_data, **kwargs)
            transformed_data = transform.transform(transformed_data)

            # Record this step in history
            self._record_history_step(transform, data.shape, transformed_data.shape)

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply all transforms in the pipeline"""
        transformed_data = data.copy()
        for transform in self.transforms:
            transformed_data = transform.transform(transformed_data)
        return transformed_data

    def fit_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Fit and then transform the data"""
        self.fit(data, **kwargs)
        return self.transform(data)

    def _record_history_step(self, transform: BaseTransform,
                             input_shape: Tuple[int, ...],
                             output_shape: Tuple[int, ...]) -> None:
        """Record a step in the transformation history"""
        step = {
            "transform": transform.serialize(),
            "timestamp": datetime.now().isoformat(),
            "input_shape": input_shape,
            "output_shape": output_shape
        }
        self.history.append(step)
        self.last_modified_timestamp = step["timestamp"]

    def serialize(self) -> Dict[str, Any]:
        """Serialize the pipeline to a dictionary"""
        return {
            "name": self.name,
            "transforms": [t.serialize() for t in self.transforms],
            "history": self.history,
            "column_mapping": self.column_mapping,
            "creation_timestamp": self.creation_timestamp,
            "last_modified_timestamp": self.last_modified_timestamp
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any], registry: Dict[str, type]) -> 'TransformPipeline':
        """Deserialize a pipeline from a dictionary"""
        pipeline = cls(name=data["name"])
        pipeline.history = data["history"]
        pipeline.column_mapping = data["column_mapping"]
        pipeline.creation_timestamp = data["creation_timestamp"]
        pipeline.last_modified_timestamp = data["last_modified_timestamp"]

        for transform_data in data["transforms"]:
            transform = BaseTransform.deserialize(transform_data, registry)
            pipeline.transforms.append(transform)

        return pipeline

    def save_to_h5(self, file_path: str, data: np.ndarray = None,
                   group_name: str = None) -> str:
        """
        Save the pipeline and optionally the transformed data to an H5 file

        Args:
            file_path: Path to the H5 file
            data: Optional data to transform and save
            group_name: Optional group name in the H5 file, defaults to pipeline name

        Returns:
            The group name where the data was stored
        """
        group_name = group_name or self.name

        with h5py.File(file_path, 'a') as f:
            # Create a group for this pipeline if it doesn't exist
            if group_name in f:
                del f[group_name]  # Replace existing group

            group = f.create_group(group_name)

            # Store pipeline metadata
            metadata_json = json.dumps(self.serialize())
            group.attrs['pipeline_metadata'] = metadata_json

            # Store the transformed data if provided
            if data is not None:
                transformed_data = self.transform(data)
                dataset = group.create_dataset('data', data=transformed_data)
                dataset.attrs['shape'] = transformed_data.shape
                dataset.attrs['columns'] = json.dumps({v: k for k, v in self.column_mapping.items()})

            # Store original column descriptions
            group.attrs['column_mapping'] = json.dumps(self.column_mapping)

        return group_name

    @classmethod
    def load_from_h5(cls, file_path: str, group_name: str,
                     registry: Dict[str, type]) -> Tuple['TransformPipeline', Optional[np.ndarray]]:
        """
        Load a pipeline and optionally the transformed data from an H5 file

        Args:
            file_path: Path to the H5 file
            group_name: Group name in the H5 file
            registry: Registry of transform classes

        Returns:
            Tuple of (pipeline, data) where data may be None if not stored
        """
        pipeline = None
        data = None

        with h5py.File(file_path, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group {group_name} not found in file {file_path}")

            group = f[group_name]

            # Load pipeline metadata
            metadata_json = group.attrs['pipeline_metadata']
            pipeline_data = json.loads(metadata_json)
            pipeline = cls.deserialize(pipeline_data, registry)

            # Load the data if it exists
            if 'data' in group:
                data = group['data'][:]

        return pipeline, data


# Registry for transforms to enable serialization/deserialization
TRANSFORM_REGISTRY = {}


def register_transform(cls):
    """Register a transform class in the global registry"""
    TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls