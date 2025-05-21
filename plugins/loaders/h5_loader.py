#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDF5 Data Loader for the ML Provenance Tracker Framework.
This module handles loading data from HDF5 files.
"""

import os
import logging
import numpy as np
import pandas as pd
import h5py
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class H5Loader:
    """
    Data loader for HDF5 files.
    
    This loader handles:
    - Loading data from HDF5 files
    - Extracting specific datasets
    - Converting to pandas DataFrames
    """
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initialize the HDF5 loader.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional parameters
        """
        self.config = config
        self.dataset_path = config.get("dataset_path")
        self.dataset_keys = config.get("dataset_keys", None)
        self.n_samples = config.get("n_samples", None)
        
        # Store column mapping if provided
        self.column_mapping = kwargs.get("column_mapping", {})
        
        # Initialize metadata
        self.metadata = {
            "loader_type": "H5Loader",
            "file_path": self.dataset_path,
            "dataset_keys": self.dataset_keys
        }
        
        logger.info(f"H5Loader initialized for file: {self.dataset_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from HDF5 file.
        
        Returns:
            Pandas DataFrame with loaded data
        """
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        logger.info(f"Loading data from {self.dataset_path}")
        
        try:
            with h5py.File(self.dataset_path, 'r') as h5file:
                # Get the dataset keys if not specified
                if self.dataset_keys is None:
                    self.dataset_keys = list(h5file.keys())
                    logger.info(f"Found datasets: {', '.join(self.dataset_keys)}")
                
                # Load data from each dataset
                data_dict = {}
                for key in self.dataset_keys:
                    if key in h5file:
                        dataset = h5file[key]
                        
                        # Apply sample limit if specified
                        if self.n_samples is not None and len(dataset) > self.n_samples:
                            data_dict[key] = dataset[:self.n_samples][()]
                        else:
                            data_dict[key] = dataset[()]
                    else:
                        logger.warning(f"Dataset key '{key}' not found in HDF5 file")
                
                # Update metadata
                self.metadata["num_datasets"] = len(data_dict)
                self.metadata["sample_sizes"] = {k: len(v) for k, v in data_dict.items()}
                
                # Convert to DataFrame
                df = self._convert_to_dataframe(data_dict)
                
                # Record total samples
                self.metadata["total_samples"] = len(df)
                
                logger.info(f"Loaded {len(df)} samples from {len(data_dict)} datasets")
                return df
        
        except Exception as e:
            logger.error(f"Error loading data from HDF5 file: {e}")
            raise
    
    def _convert_to_dataframe(self, data_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Convert dictionary of arrays to pandas DataFrame.
        
        Args:
            data_dict: Dictionary of dataset name to numpy array
            
        Returns:
            Pandas DataFrame with combined data
        """
        # Check if arrays have the same length
        lengths = [len(arr) for arr in data_dict.values()]
        if len(set(lengths)) > 1:
            logger.warning(f"Datasets have different lengths: {lengths}")
            
            # Find minimum length to truncate arrays
            min_length = min(lengths)
            logger.info(f"Truncating arrays to minimum length: {min_length}")
            
            # Truncate arrays
            data_dict = {k: v[:min_length] for k, v in data_dict.items()}
        
        # Convert to DataFrame
        df = pd.DataFrame()
        
        # Map dataset keys to column names if mapping exists
        for key, array in data_dict.items():
            # Get column name from mapping or use key as is
            column_name = self.column_mapping.get(key, key)
            
            # Reshape array if needed (1D arrays are treated as columns)
            if len(array.shape) == 1:
                df[column_name] = array
            elif len(array.shape) == 2:
                # For 2D arrays, create multiple columns with suffixes
                for i in range(array.shape[1]):
                    df[f"{column_name}_{i}"] = array[:, i]
            else:
                logger.warning(f"Skipping dataset '{key}' with shape {array.shape} (too many dimensions)")
        
        return df
