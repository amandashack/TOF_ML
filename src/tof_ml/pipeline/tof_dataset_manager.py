# src/tof_ml/pipeline/tof_dataset_manager.py
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
import pandas as pd

from src.tof_ml.pipeline.tof_dataset import TOFDataset
from src.tof_ml.pipeline.batch_processor import DatasetVariationKey
from src.tof_ml.transforms.transform_pipeline import TransformPipeline, TRANSFORM_REGISTRY

logger = logging.getLogger("tof_dataset_manager")


class TOFDatasetManager:
    """
    Manages multiple TOF datasets across different processed files.
    Provides utilities for discovery, filtering, and loading of datasets.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the dataset manager.

        Args:
            data_dir: Directory containing processed H5 files
        """
        self.data_dir = data_dir
        self.file_info = []
        self.catalog = pd.DataFrame()

        # Scan for datasets
        self.scan_datasets()

    def scan_datasets(self):
        """Scan the data directory for H5 files with TOF data"""
        h5_files = glob.glob(os.path.join(self.data_dir, "*.h5"))

        file_info = []
        for file_path in h5_files:
            try:
                info = self._extract_file_info(file_path)
                if info:
                    file_info.append(info)
            except Exception as e:
                logger.warning(f"Error analyzing file {file_path}: {e}")

        self.file_info = file_info

        # Convert to DataFrame for easier filtering
        if file_info:
            self.catalog = pd.DataFrame(file_info)
            logger.info(f"Found {len(self.catalog)} TOF datasets")
        else:
            logger.warning(f"No valid TOF datasets found in {self.data_dir}")

    def _extract_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata information from an H5 file.

        Args:
            file_path: Path to the H5 file

        Returns:
            Dictionary with file information or None if not a valid TOF dataset
        """
        try:
            with h5py.File(file_path, 'r') as f:
                # Check if this is a TOF dataset file
                if 'variation_key' not in f.attrs:
                    return None

                # Extract basic file info
                file_info = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'creation_timestamp': f.attrs.get('creation_timestamp', ''),
                    'variation_key': int(f.attrs.get('variation_key', 0)),
                    'config_hash': f.attrs.get('config_hash', ''),
                    'total_samples': int(f.attrs.get('total_samples', 0)),
                    'pos_samples': int(f.attrs.get('pos_retardation_samples', 0)),
                    'neg_samples': int(f.attrs.get('neg_retardation_samples', 0))
                }

                # Add variation flags
                variation_flags = DatasetVariationKey.decode_key(file_info['variation_key'])
                file_info.update(variation_flags)

                # Check for groups
                file_info['has_pos_group'] = 'pos_retardation' in f
                file_info['has_neg_group'] = 'neg_retardation' in f

                # Get available column info
                if 'column_info' in f.attrs:
                    column_info = json.loads(f.attrs['column_info'])
                    file_info['columns'] = column_info.get('names', [])

                # Sample format (shape, dtype)
                if 'data' in f:
                    file_info['data_shape'] = str(f['data'].shape)
                    file_info['data_dtype'] = str(f['data'].dtype)
                elif 'pos_retardation' in f and 'data' in f['pos_retardation']:
                    file_info['data_shape'] = str(f['pos_retardation']['data'].shape)
                    file_info['data_dtype'] = str(f['pos_retardation']['data'].dtype)

                return file_info
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return None

    def filter_datasets(self, **kwargs) -> pd.DataFrame:
        """
        Filter datasets based on specified criteria.

        Args:
            **kwargs: Key-value pairs for filtering

        Returns:
            Filtered DataFrame of datasets
        """
        if self.catalog.empty:
            logger.warning("No datasets available to filter")
            return pd.DataFrame()

        query_parts = []
        for key, value in kwargs.items():
            if key in self.catalog.columns:
                if isinstance(value, bool):
                    query_parts.append(f"{key} == {value}")
                elif isinstance(value, (int, float)):
                    query_parts.append(f"{key} == {value}")
                elif isinstance(value, str):
                    query_parts.append(f"{key} == '{value}'")
                elif isinstance(value, (list, tuple)):
                    values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                    query_parts.append(f"{key} in [{values_str}]")

        if not query_parts:
            return self.catalog

        query = " and ".join(query_parts)
        filtered = self.catalog.query(query)
        logger.info(f"Filter query '{query}' returned {len(filtered)} datasets")
        return filtered

    def load_dataset(self, file_path: str,
                     group_path: str = 'data',
                     batch_size: int = 32,
                     feature_columns: Optional[List[str]] = None,
                     label_columns: Optional[List[str]] = None) -> TOFDataset:
        """
        Load a TOF dataset from a file.

        Args:
            file_path: Path to the H5 file
            group_path: Path to the group within the H5 file
            batch_size: Batch size
            feature_columns: Column names or indices for features
            label_columns: Column names or indices for labels

        Returns:
            TOFDataset instance
        """
        return TOFDataset.load_from_processed_file(
            file_path=file_path,
            group_path=group_path,
            batch_size=batch_size,
            feature_columns=feature_columns,
            label_columns=label_columns
        )

    def load_dataset_by_index(self, index: int,
                              group: str = 'pos_retardation',
                              batch_size: int = 32,
                              feature_columns: Optional[List[str]] = None,
                              label_columns: Optional[List[str]] = None) -> TOFDataset:
        """
        Load a TOF dataset by its index in the catalog.

        Args:
            index: Index in the catalog
            group: Group to load ('pos_retardation', 'neg_retardation', or 'data')
            batch_size: Batch size
            feature_columns: Column names or indices for features
            label_columns: Column names or indices for labels

        Returns:
            TOFDataset instance
        """
        if self.catalog.empty or index >= len(self.catalog):
            raise ValueError(f"Invalid dataset index: {index}")

        row = self.catalog.iloc[index]
        file_path = row['file_path']

        # Determine the group path
        if group == 'pos_retardation' and row['has_pos_group']:
            group_path = 'pos_retardation/data'
        elif group == 'neg_retardation' and row['has_neg_group']:
            group_path = 'neg_retardation/data'
        else:
            group_path = 'data'

        return self.load_dataset(
            file_path=file_path,
            group_path=group_path,
            batch_size=batch_size,
            feature_columns=feature_columns,
            label_columns=label_columns
        )

    def summary(self):
        """
        Print a summary of available datasets.
        """
        if self.catalog.empty:
            print("No datasets found.")
            return

        print(f"Found {len(self.catalog)} TOF datasets:")
        print("\nDataset variations:")

        # Count datasets by variation type
        variation_counts = {}
        for _, row in self.catalog.iterrows():
            variation_desc = DatasetVariationKey.key_to_string(row['variation_key'])
            if variation_desc not in variation_counts:
                variation_counts[variation_desc] = 0
            variation_counts[variation_desc] += 1

        for desc, count in sorted(variation_counts.items()):
            print(f"  - {desc}: {count} dataset(s)")

        print("\nSample counts:")
        total_samples = self.catalog['total_samples'].sum()
        pos_samples = self.catalog['pos_samples'].sum()
        neg_samples = self.catalog['neg_samples'].sum()
        print(f"  - Total samples: {total_samples}")
        print(f"  - Positive retardation samples: {pos_samples}")
        print(f"  - Negative retardation samples: {neg_samples}")

        print("\nMost recent datasets:")
        if 'creation_timestamp' in self.catalog.columns:
            recent = self.catalog.sort_values('creation_timestamp', ascending=False).head(5)
            for idx, row in recent.iterrows():
                print(f"  - {row['file_name']} ({row['creation_timestamp']})")

        print("\nFor detailed information, use the catalog attribute.")

    def get_dataset_info(self, index: int) -> Dict[str, Any]:
        """
        Get detailed information about a dataset.

        Args:
            index: Index in the catalog

        Returns:
            Dictionary with dataset information
        """
        if self.catalog.empty or index >= len(self.catalog):
            raise ValueError(f"Invalid dataset index: {index}")

        row = self.catalog.iloc[index].to_dict()

        # Add a human-readable variation description
        row['variation_description'] = DatasetVariationKey.key_to_string(row['variation_key'])

        return row