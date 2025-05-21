#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced HDF5 Data Loader for the ML Provenance Tracker Framework.
This module handles loading data from multiple HDF5 files with complex structure.
"""

import os
import logging
import glob
import re
import numpy as np
import h5py
from typing import Dict, Any, Optional, Union, List, Tuple, Iterator

# Import the base classes
from src.tof_ml.data.base_data_loader import FileBasedDataLoader, BatchedDataLoader

logger = logging.getLogger(__name__)

# Constants for column mappings
COLUMN_MAPPING = {
    'kinetic_energy': 0,
    'elevation': 1,
    'mid1': 2,
    'mid2': 3,
    'retardation': 4,
    'tof': 5,
    'x_pos': 6,
    'y_pos': 7
}

REDUCED_COLUMN_MAPPING = {
    'kinetic_energy': 0,
    'tof': 1,
    'x_pos': 2,
    'y_pos': 3
}


class EnhancedH5Loader(BatchedDataLoader, FileBasedDataLoader):
    """
    Enhanced data loader for HDF5 files.

    This loader handles:
    - Loading data from multiple HDF5 files
    - Processing directory structures with retardation values
    - Batch loading for large datasets
    - Converting to numpy arrays (not pandas)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the enhanced HDF5 loader.

        Args:
            config: Configuration dictionary
            **kwargs: Additional parameters including:
                - mid1: Mid1 voltage configuration
                - mid2: Mid2 voltage configuration
                - column_mapping: Mapping of column names to indices
        """
        super().__init__(config, **kwargs)

        # Extract H5-specific parameters
        self.mid1 = kwargs.get("mid1", self.config.get("mid1", [0.0, 0.0]))
        self.mid2 = kwargs.get("mid2", self.config.get("mid2", [0.0, 0.0]))

        # Column mapping - prefer from kwargs, then config, then default
        self.column_mapping = kwargs.get("column_mapping",
                                         self.config.get("column_mapping", COLUMN_MAPPING))
        self.reduced_column_mapping = kwargs.get("reduced_column_mapping",
                                                 self.config.get("reduced_column_mapping", REDUCED_COLUMN_MAPPING))

        # Pattern for extracting information from filenames
        self.file_pattern = r'sim_(?P<r_sign>neg|pos)_R(?P<r_value>\d+)_pos_(?P<mid1>\d+\.\d+)_pos_(?P<mid2>\d+\.\d+)_(?P<energy>\d+)\.h5'

        # Cache for file paths
        self._file_paths = None
        self._file_info = {}

        # Update metadata
        self.metadata.update({
            "mid1": self.mid1,
            "mid2": self.mid2,
            "column_mapping": self.column_mapping,
            "file_pattern": self.file_pattern
        })

        logger.info(f"EnhancedH5Loader initialized for path: {self.dataset_path}")

    def _initialize_file_discovery(self) -> List[str]:
        """
        Discover and cache file paths.

        Returns:
            List of file paths to load
        """
        if self._file_paths is not None:
            return self._file_paths

        # Find all retardation directories
        if os.path.isdir(self.dataset_path):
            r_dirs = self._find_retardation_dirs()
            file_paths = self._find_h5_files(r_dirs)
        else:
            # Single file case
            file_paths = [self.dataset_path]

        # Filter files based on mid1 and mid2 if provided
        if self.mid1 != [0.0, 0.0] or self.mid2 != [0.0, 0.0]:
            file_paths = self._filter_files_by_voltage(file_paths)

        # Cache results
        self._file_paths = file_paths

        # Update metadata
        self.metadata["file_count"] = len(file_paths)

        logger.info(f"Discovered {len(file_paths)} H5 files")
        return file_paths

    def load_data(self) -> np.ndarray:
        """
        Load data from multiple HDF5 files.

        Returns:
            Numpy array with the loaded data
        """
        logger.info(f"Loading data from {self.dataset_path}")

        # Discover files
        file_paths = self._initialize_file_discovery()

        if not file_paths:
            raise ValueError(f"No HDF5 files found matching the criteria in {self.dataset_path}")

        # Load data from files
        if self.n_samples and self.n_samples < self.batch_size:
            # Small dataset case - load all at once
            data = self._load_files(file_paths, self.n_samples)
        else:
            # Large dataset case - load in batches
            data = self._load_files_in_batches(file_paths)

        # Cache the data
        self._data_cache = data
        self._data_loaded = True

        # Update metadata
        self.metadata["sample_count"] = len(data)

        logger.info(f"Loaded {len(data)} samples from {len(file_paths)} files")
        return data

    def _get_batch_count(self) -> int:
        """
        Get the total number of batches available.

        Returns:
            Total number of batches
        """
        file_paths = self._initialize_file_discovery()

        # Calculate total samples across all files
        total_samples = 0
        for file_path in file_paths:
            file_info = self._get_file_info(file_path)
            total_samples += file_info.get('sample_count', 0)

        # Apply sample limit if specified
        if self.n_samples is not None and self.n_samples < total_samples:
            total_samples = self.n_samples

        # Calculate number of batches
        if self.batch_size:
            return (total_samples + self.batch_size - 1) // self.batch_size
        else:
            return 1

    def _load_batch(self, batch_index: int) -> np.ndarray:
        """
        Load a specific batch of data.

        Args:
            batch_index: Index of the batch to load

        Returns:
            Batch of data
        """
        file_paths = self._initialize_file_discovery()

        # Calculate which files and offsets correspond to this batch
        start_sample = batch_index * self.batch_size
        end_sample = start_sample + self.batch_size

        # Find the right files for this batch
        current_sample = 0
        batch_data = []

        for file_path in file_paths:
            file_info = self._get_file_info(file_path)
            file_samples = file_info.get('sample_count', 0)

            # Check if this file contains samples for our batch
            if current_sample + file_samples > start_sample:
                # Calculate the portion of this file we need
                file_start = max(0, start_sample - current_sample)
                file_end = min(file_samples, end_sample - current_sample)

                # Load the required portion
                file_data = self._read_h5_file(file_path)
                batch_data.append(file_data[file_start:file_end])

                # Check if we've loaded enough
                if current_sample + file_samples >= end_sample:
                    break

            current_sample += file_samples

        # Combine batch data
        if batch_data:
            return np.vstack(batch_data)
        else:
            return np.array([])

    def _find_retardation_dirs(self) -> List[str]:
        """Find all retardation directories in the dataset path."""
        r_dirs = []

        # Look for directories named 'R*'
        for item in os.listdir(self.dataset_path):
            item_path = os.path.join(self.dataset_path, item)
            if os.path.isdir(item_path) and item.startswith('R'):
                r_dirs.append(item_path)

        logger.info(f"Found {len(r_dirs)} retardation directories")
        return r_dirs

    def _find_h5_files(self, r_dirs: List[str]) -> List[str]:
        """Find all HDF5 files in the retardation directories."""
        file_paths = []

        for r_dir in r_dirs:
            h5_files = glob.glob(os.path.join(r_dir, "*.h5"))
            file_paths.extend(h5_files)

        logger.info(f"Found {len(file_paths)} H5 files")
        return file_paths

    def _filter_files_by_voltage(self, file_paths: List[str]) -> List[str]:
        """Filter files by mid1 and mid2 voltage configurations."""
        filtered_paths = []

        for path in file_paths:
            match = re.search(self.file_pattern, os.path.basename(path))
            if match:
                mid1_val = float(match.group('mid1'))
                mid2_val = float(match.group('mid2'))

                # Check if this file matches the desired voltage configuration
                if (mid1_val == self.mid1[0] or mid1_val == self.mid1[1]) and \
                        (mid2_val == self.mid2[0] or mid2_val == self.mid2[1]):
                    filtered_paths.append(path)

        logger.info(f"Filtered to {len(filtered_paths)} files matching voltage configuration")
        return filtered_paths

    def _get_file_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get cached information about a file.

        Args:
            filepath: Path to the H5 file

        Returns:
            Dictionary with file information
        """
        if filepath in self._file_info:
            return self._file_info[filepath]

        # Extract basic file info
        file_info = {
            'path': filepath,
            'basename': os.path.basename(filepath),
            'retardation': self._parse_retardation_from_filename(filepath),
            'mid1': 0.0,
            'mid2': 0.0,
            'energy': 0.0,
            'sample_count': 0
        }

        # Extract info from filename
        match = re.search(self.file_pattern, os.path.basename(filepath))
        if match:
            file_info['mid1'] = float(match.group('mid1'))
            file_info['mid2'] = float(match.group('mid2'))
            file_info['energy'] = float(match.group('energy'))

        # Get sample count by opening the file
        try:
            with h5py.File(filepath, 'r') as f:
                if 'data1' in f and 'initial_ke' in f['data1']:
                    file_info['sample_count'] = len(f['data1']['initial_ke'])
        except Exception as e:
            logger.warning(f"Error getting sample count for {filepath}: {e}")

        # Cache and return
        self._file_info[filepath] = file_info
        return file_info

    def _parse_retardation_from_filename(self, filepath: str) -> float:
        """
        Parse retardation value from filename.

        Args:
            filepath: Path to the H5 file

        Returns:
            Retardation value
        """
        match = re.search(self.file_pattern, os.path.basename(filepath))
        if match:
            r_sign = match.group('r_sign')
            r_value = float(match.group('r_value'))
            return -r_value if r_sign == 'neg' else r_value

        return 0.0

    def _read_h5_file(self, filepath: str) -> np.ndarray:
        """
        Opens an .h5 file and returns data array.

        Args:
            filepath: Path to the H5 file

        Returns:
            Numpy array with the loaded data
        """
        # Get file info for cached values
        file_info = self._get_file_info(filepath)

        try:
            with h5py.File(filepath, 'r') as f:
                if 'data1' not in f:
                    logger.error(f"No 'data1' group found in {filepath}")
                    return np.array([])

                data_group = f['data1']

                # Extract fields based on column mapping
                field_data = {}

                # Map standard field names to H5 dataset names
                field_mapping = {
                    'initial_ke': ['initial_ke'],
                    'kinetic_energy': ['initial_ke'],
                    'elevation': ['initial_elevation'],
                    'initial_elevation': ['initial_elevation'],
                    'tof': ['tof'],
                    'tof_values': ['tof'],
                    'x': ['x'],
                    'x_tof': ['x'],
                    'x_pos': ['x'],
                    'y': ['y'],
                    'y_tof': ['y'],
                    'y_pos': ['y']
                }

                # Extract available fields from the file
                for column_name, dataset_names in field_mapping.items():
                    for dataset_name in dataset_names:
                        if dataset_name in data_group:
                            field_data[column_name] = data_group[dataset_name][:]
                            break

                # Check if any data was found
                if not field_data or len(field_data.get('initial_ke', [])) == 0:
                    logger.warning(f"No data found in {filepath}")
                    return np.array([])

                # Determine the number of samples
                N = len(field_data['initial_ke'])

                # Create arrays for mid1, mid2, and retardation
                field_data['mid1'] = np.full((N,), file_info['mid1'])
                field_data['mid2'] = np.full((N,), file_info['mid2'])
                field_data['retardation'] = np.full((N,), file_info['retardation'])

                # Organize data according to column mapping
                data_list = []

                for i in range(8):  # Assuming 8 columns
                    # Find column name that maps to this index
                    col_names = [k for k, v in self.column_mapping.items() if v == i]

                    if col_names and col_names[0] in field_data:
                        data_list.append(field_data[col_names[0]])
                    else:
                        # Use a default if the column is not found
                        logger.warning(f"Column index {i} not found in data, using zeros")
                        data_list.append(np.zeros(N))

                # Combine all data into a single array
                data_array = np.column_stack(data_list)
                return data_array

        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            return np.array([])

    def _load_files(self, file_paths: List[str], max_samples: Optional[int] = None) -> np.ndarray:
        """
        Load data from files up to max_samples.

        Args:
            file_paths: List of file paths to load
            max_samples: Maximum number of samples to load

        Returns:
            Numpy array with the loaded data
        """
        data_chunks = []
        samples_loaded = 0

        for file_path in file_paths:
            try:
                # Read data from H5 file
                data = self._read_h5_file(file_path)

                if data.size == 0:
                    continue

                # Determine how many samples to load
                if max_samples is not None:
                    samples_to_load = min(len(data), max_samples - samples_loaded)
                    if samples_to_load <= 0:
                        break
                    data = data[:samples_to_load]

                data_chunks.append(data)
                samples_loaded += len(data)

                logger.debug(f"Loaded {len(data)} samples from {os.path.basename(file_path)}")

            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")

        if not data_chunks:
            raise ValueError("No data loaded from any file")

        # Combine all chunks
        return np.vstack(data_chunks)

    def _load_files_in_batches(self, file_paths: List[str]) -> np.ndarray:
        """
        Load data from files in batches.

        Args:
            file_paths: List of file paths to load

        Returns:
            Numpy array with the loaded data
        """
        # Calculate total samples and create output array
        total_samples = 0
        for file_path in file_paths:
            file_info = self._get_file_info(file_path)
            total_samples += file_info['sample_count']

        # Apply sample limit if specified
        if self.n_samples is not None and self.n_samples < total_samples:
            total_samples = self.n_samples

        # Check if we found any samples
        if total_samples == 0:
            raise ValueError("No samples found in any file")

        # Create output array (8 columns standard)
        output_data = np.zeros((total_samples, 8))

        # Load data in batches
        samples_loaded = 0
        samples_remaining = total_samples

        for file_path in file_paths:
            if samples_remaining <= 0:
                break

            file_info = self._get_file_info(file_path)
            if file_info['sample_count'] == 0:
                continue

            try:
                # Read this file's data
                data = self._read_h5_file(file_path)

                if data.size == 0:
                    continue

                # Determine how many samples to load from this file
                file_samples = min(len(data), samples_remaining)
                data = data[:file_samples]

                # Store in output array
                output_data[samples_loaded:samples_loaded + file_samples] = data

                samples_loaded += file_samples
                samples_remaining -= file_samples

                logger.debug(f"Loaded {file_samples} samples from {os.path.basename(file_path)}")

            except Exception as e:
                logger.error(f"Error loading file {file_path} in batches: {e}")

        return output_data[:samples_loaded]  # Return only the filled portion

    def _extract_column_names(self) -> List[str]:
        """
        Extract column names in the order they appear in the data.

        Returns:
            List of column names
        """
        # Sort column mapping by index to get ordered column names
        sorted_columns = sorted(self.column_mapping.items(), key=lambda x: x[1])
        return [col_name for col_name, _ in sorted_columns]

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loader and loaded data.

        Returns:
            Dictionary containing loader metadata
        """
        metadata = super().get_metadata()

        # Add H5-specific metadata
        metadata.update({
            "voltage_config": {
                "mid1": self.mid1,
                "mid2": self.mid2
            },
            "column_mapping": self.column_mapping,
            "reduced_column_mapping": self.reduced_column_mapping,
            "file_pattern": self.file_pattern
        })

        # Add file information if available
        if self._file_paths is not None:
            metadata["files_discovered"] = len(self._file_paths)

            # Add sample of file info
            if self._file_info:
                sample_files = list(self._file_info.values())[:5]  # First 5 files
                metadata["sample_file_info"] = sample_files

        return metadata