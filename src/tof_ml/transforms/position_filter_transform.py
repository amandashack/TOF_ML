#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Position Filter Transform for ToF ML.
This module handles filtering data based on position constraints.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union

from src.tof_ml.transforms.base_transform import BaseTransform
from plugins import COLUMN_MAPPING

logger = logging.getLogger(__name__)

class PositionFilterTransform(BaseTransform):
    """
    Filter data based on position constraints.
    """
    
    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        """
        Initialize the position filter transformer.
        
        Args:
            config: Configuration dictionary with position constraints
            **kwargs: Additional parameters
        """
        super().__init__(name=kwargs.get('name', 'PositionFilter'), **kwargs)
        
        # Handle both the case when config is passed directly or within kwargs
        if config is None:
            config = kwargs
        
        self.x_column = config.get("x_column", "x")
        self.y_column = config.get("y_column", "y")
        
        # Position constraints
        self.x_min = config.get("x_min", None)
        self.x_max = config.get("x_max", None)
        self.y_min = config.get("y_min", None)
        self.y_max = config.get("y_max", None)
        
        # Column mapping from raw data array indices
        self.column_mapping = kwargs.get("column_mapping", COLUMN_MAPPING)
        
        # Store metadata
        self._metadata.update({
            "transformer_type": "PositionFilterTransform",
            "x_column": self.x_column,
            "y_column": self.y_column,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max
        })
        
        logger.info(f"PositionFilterTransform initialized with x constraints: [{self.x_min}, {self.x_max}], "
                   f"y constraints: [{self.y_min}, {self.y_max}]")
    
    def fit(self, data: np.ndarray, **kwargs) -> 'PositionFilterTransform':
        """
        No fitting required for this transform, just records statistics.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Self for chaining
        """
        self._is_fitted = True
        
        # Extract x and y column indices or names
        x_idx = self._get_column_index(self.x_column, data)
        y_idx = self._get_column_index(self.y_column, data)
        
        # Record statistics about the position columns
        if isinstance(data, np.ndarray) and len(data.shape) == 2:
            self._metadata["x_min_value"] = float(np.min(data[:, x_idx]))
            self._metadata["x_max_value"] = float(np.max(data[:, x_idx]))
            self._metadata["y_min_value"] = float(np.min(data[:, y_idx]))
            self._metadata["y_max_value"] = float(np.max(data[:, y_idx]))
        elif isinstance(data, pd.DataFrame):
            self._metadata["x_min_value"] = float(data[self.x_column].min())
            self._metadata["x_max_value"] = float(data[self.x_column].max())
            self._metadata["y_min_value"] = float(data[self.y_column].min())
            self._metadata["y_max_value"] = float(data[self.y_column].max())
        
        return self
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Filter data based on position constraints.
        
        Args:
            data: Input data as DataFrame or numpy array
            
        Returns:
            Filtered data
        """
        logger.info("Applying position filter transform")
        
        # Get original data shape
        if isinstance(data, pd.DataFrame):
            original_shape = data.shape
            
            # Find columns for x and y
            x_col = self.x_column if self.x_column in data.columns else None
            y_col = self.y_column if self.y_column in data.columns else None
            
            # Create mask
            mask = pd.Series(True, index=data.index)
            
            if x_col is not None and self.x_min is not None:
                mask &= data[x_col] > self.x_min
            if x_col is not None and self.x_max is not None:
                mask &= data[x_col] < self.x_max
            if y_col is not None and self.y_min is not None:
                mask &= data[y_col] > self.y_min
            if y_col is not None and self.y_max is not None:
                mask &= data[y_col] < self.y_max
            
            # Apply mask
            filtered_data = data[mask]
            
        else:  # Numpy array
            original_shape = data.shape
            
            # Find the column indices for x and y in the raw array
            x_idx = self._get_column_index(self.x_column, data)
            y_idx = self._get_column_index(self.y_column, data)
            
            # Create mask
            mask = np.ones(data.shape[0], dtype=bool)
            
            if self.x_min is not None:
                mask &= data[:, x_idx] > self.x_min
            if self.x_max is not None:
                mask &= data[:, x_idx] < self.x_max
            if self.y_min is not None:
                mask &= data[:, y_idx] > self.y_min
            if self.y_max is not None:
                mask &= data[:, y_idx] < self.y_max
            
            # Apply mask
            filtered_data = data[mask]
        
        # Log the filtering results
        logger.info(f"Position filtering: {original_shape[0]} -> {filtered_data.shape[0]} samples "
                   f"(removed {original_shape[0] - filtered_data.shape[0]} samples)")
        
        # Update metadata
        self._metadata["input_samples"] = original_shape[0]
        self._metadata["output_samples"] = filtered_data.shape[0]
        self._metadata["filtered_samples"] = original_shape[0] - filtered_data.shape[0]
        self._metadata["filter_percentage"] = round((original_shape[0] - filtered_data.shape[0]) / original_shape[0] * 100, 2)
        
        return filtered_data
    
    def _get_column_index(self, column_name: str, data: Union[pd.DataFrame, np.ndarray]) -> int:
        """
        Get the column index for a given column name.
        
        Args:
            column_name: Column name
            data: Input data
            
        Returns:
            Column index
        """
        if isinstance(data, pd.DataFrame):
            if column_name in data.columns:
                return data.columns.get_loc(column_name)
            else:
                # Try to find column using mapping
                for key, value in self.column_mapping.items():
                    if key.startswith(column_name) and value in data.columns:
                        return data.columns.get_loc(value)
                
                raise ValueError(f"Column '{column_name}' not found in DataFrame")
        else:
            # For numpy array, use column mapping
            # Find exact match
            if column_name in self.column_mapping:
                return self.column_mapping[column_name]
            
            # Find partial match (e.g., 'x' matches 'x_tof')
            for key, value in self.column_mapping.items():
                if key.startswith(column_name):
                    return value
            
            # Default values for common columns if not found
            if column_name == "x" or column_name.startswith("x_"):
                return 6  # Default x position column
            elif column_name == "y" or column_name.startswith("y_"):
                return 7  # Default y position column
            
            raise ValueError(f"Column '{column_name}' not found in column mapping")
