# src/tof_ml/pipeline/batch_processor.py
from typing import Dict, List, Tuple, Optional, Union, Callable, Iterator, Any
import os
import numpy as np
import h5py
import logging
import yaml
import json
import time
from datetime import datetime
import hashlib
from tqdm import tqdm
import multiprocessing as mp
from queue import Queue
from threading import Thread

from src.tof_ml.data.enhanced_data_loader import EnhancedDataLoader
from src.tof_ml.transforms.transform_pipeline import TransformPipeline
from src.tof_ml.transforms.common_transforms import *
from src.tof_ml.data.enhanced_h5_data_loader import EnhancedH5DataLoader
from src.tof_ml.data.nm_csv_data_loader import NMCsvDataLoader
from src.tof_ml.utils.json_utils import json_dump, json_dumps

logger = logging.getLogger("batch_processor")


class DatasetVariationKey:
    """
    Manages the 32-bit signed integer key that represents what varies in a dataset.

    Bit positions (0-indexed):
    - Bits 0-20: Reserved for other voltage values
    - Bit 21: Mid1 voltage (voltage 22)
    - Bit 24: Mid2 voltage (voltage 25)
    - Bit 25: Retardation
    - Bit 26: Pass energy
    - Bit 27: Elevation
    - Bit 28: X origin position
    - Bit 29: Y origin position
    - Bit 30: Flight tube length
    - Bit 31: Sign of retardation (0=positive only, 1=includes negative)
    """

    # Define bit positions
    MID1_BIT = 21
    MID2_BIT = 24
    RETARDATION_BIT = 25
    PASS_ENERGY_BIT = 26
    ELEVATION_BIT = 27
    X_ORIGIN_BIT = 28
    Y_ORIGIN_BIT = 29
    FLIGHT_TUBE_BIT = 30
    SIGN_BIT = 31

    @staticmethod
    def create_key(mid1_varies: bool = False,
                   mid2_varies: bool = False,
                   retardation_varies: bool = False,
                   pass_energy_varies: bool = False,
                   elevation_varies: bool = False,
                   x_origin_varies: bool = False,
                   y_origin_varies: bool = False,
                   flight_tube_varies: bool = False,
                   includes_negative_retardation: bool = False) -> int:
        """
        Create a 32-bit signed integer key representing what varies in the dataset.

        Args:
            mid1_varies: Whether mid1 voltage varies
            mid2_varies: Whether mid2 voltage varies
            retardation_varies: Whether retardation varies
            pass_energy_varies: Whether pass energy varies
            elevation_varies: Whether elevation varies
            x_origin_varies: Whether x origin position varies
            y_origin_varies: Whether y origin position varies
            flight_tube_varies: Whether flight tube length varies
            includes_negative_retardation: Whether dataset includes negative retardation

        Returns:
            32-bit signed integer representing the variations
        """
        key = 0

        if mid1_varies:
            key |= (1 << DatasetVariationKey.MID1_BIT)
        if mid2_varies:
            key |= (1 << DatasetVariationKey.MID2_BIT)
        if retardation_varies:
            key |= (1 << DatasetVariationKey.RETARDATION_BIT)
        if pass_energy_varies:
            key |= (1 << DatasetVariationKey.PASS_ENERGY_BIT)
        if elevation_varies:
            key |= (1 << DatasetVariationKey.ELEVATION_BIT)
        if x_origin_varies:
            key |= (1 << DatasetVariationKey.X_ORIGIN_BIT)
        if y_origin_varies:
            key |= (1 << DatasetVariationKey.Y_ORIGIN_BIT)
        if flight_tube_varies:
            key |= (1 << DatasetVariationKey.FLIGHT_TUBE_BIT)
        if includes_negative_retardation:
            key |= (1 << DatasetVariationKey.SIGN_BIT)

        return key

    @staticmethod
    def decode_key(key: int) -> Dict[str, bool]:
        """
        Decode a key into its component variations.

        Args:
            key: 32-bit signed integer key

        Returns:
            Dictionary with boolean values for each variation
        """
        return {
            'mid1_varies': bool(key & (1 << DatasetVariationKey.MID1_BIT)),
            'mid2_varies': bool(key & (1 << DatasetVariationKey.MID2_BIT)),
            'retardation_varies': bool(key & (1 << DatasetVariationKey.RETARDATION_BIT)),
            'pass_energy_varies': bool(key & (1 << DatasetVariationKey.PASS_ENERGY_BIT)),
            'elevation_varies': bool(key & (1 << DatasetVariationKey.ELEVATION_BIT)),
            'x_origin_varies': bool(key & (1 << DatasetVariationKey.X_ORIGIN_BIT)),
            'y_origin_varies': bool(key & (1 << DatasetVariationKey.Y_ORIGIN_BIT)),
            'flight_tube_varies': bool(key & (1 << DatasetVariationKey.FLIGHT_TUBE_BIT)),
            'includes_negative_retardation': bool(key & (1 << DatasetVariationKey.SIGN_BIT))
        }

    @staticmethod
    def key_to_string(key: int) -> str:
        """
        Convert a key to a human-readable string.

        Args:
            key: 32-bit signed integer key

        Returns:
            Human-readable description of the key
        """
        variations = DatasetVariationKey.decode_key(key)
        parts = []

        if variations['mid1_varies']:
            parts.append("Mid1")
        if variations['mid2_varies']:
            parts.append("Mid2")
        if variations['retardation_varies']:
            parts.append("Retardation")
        if variations['pass_energy_varies']:
            parts.append("PassEnergy")
        if variations['elevation_varies']:
            parts.append("Elevation")
        if variations['x_origin_varies']:
            parts.append("X-Origin")
        if variations['y_origin_varies']:
            parts.append("Y-Origin")
        if variations['flight_tube_varies']:
            parts.append("FlightTube")

        if not parts:
            return "NoVariation"

        sign_info = "+/-" if variations['includes_negative_retardation'] else "+"
        return f"Varies({', '.join(parts)}){sign_info}"

    @staticmethod
    def analyze_data_variations(data: np.ndarray, column_mapping: Dict[str, int]) -> int:
        """
        Analyze the data to determine what varies and create a key.

        Args:
            data: The input data array
            column_mapping: Mapping from column names to indices

        Returns:
            32-bit signed integer key representing the variations
        """

        def column_varies(column_name: str) -> bool:
            if column_name not in column_mapping:
                return False
            col_idx = column_mapping[column_name]
            if col_idx >= data.shape[1]:
                return False
            # Check if column has more than one unique value, accounting for floating point
            unique_vals = np.unique(np.round(data[:, col_idx], decimals=6))
            return len(unique_vals) > 1

        # Check for negative retardation
        has_negative_retardation = False
        if 'retardation' in column_mapping:
            ret_idx = column_mapping['retardation']
            if ret_idx < data.shape[1]:
                min_ret = np.min(data[:, ret_idx])
                has_negative_retardation = min_ret < 0

        # Create the key
        return DatasetVariationKey.create_key(
            mid1_varies=column_varies('mid1_ratio'),
            mid2_varies=column_varies('mid2_ratio'),
            retardation_varies=column_varies('retardation'),
            pass_energy_varies=column_varies('initial_ke'),  # Assuming pass energy is initial_ke
            elevation_varies=column_varies('initial_elevation'),
            x_origin_varies=column_varies('x_tof'),
            y_origin_varies=column_varies('y_tof'),
            includes_negative_retardation=has_negative_retardation
        )


class BatchDataProcessor:
    """
    Processes data in batches, applies transformations, and writes to H5 files.
    Supports parallel processing through multiple worker processes.
    """

    def __init__(self, config: Dict[str, Any],
                 num_workers: int = 1,
                 batch_size: int = 10000,
                 max_batches: Optional[int] = None):
        """
        Initialize the batch processor.

        Args:
            config: Configuration dictionary
            num_workers: Number of worker processes for parallel processing
            batch_size: Number of samples to process in each batch
            max_batches: Maximum number of batches to process (None for all)
        """
        self.config = config
        self.num_workers = min(num_workers, mp.cpu_count())
        self.batch_size = batch_size
        self.max_batches = max_batches

        # Extract relevant config sections
        self.data_config = config.get('data', {})
        self.preproc_config = config.get('preprocessing', {})
        self.loader_config_key = self.data_config.get('loader_config_key')

        # Set up paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = config.get('output_dir', f'./output/processed_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging directory for provenance tracking
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup logging to file
        log_file = os.path.join(self.log_dir, f'processing_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Hash of the config for tracking
        config_str = json.dumps(config, sort_keys=True)
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Save the full config to the log directory
        with open(os.path.join(self.log_dir, f'config_{self.config_hash}.yaml'), 'w') as f:
            yaml.dump(config, f)

        # Storage for data statistics
        self.data_stats = {}

        # Initialize the appropriate data loader
        self._setup_data_loader()

        # Initialize the transformation pipeline
        self._setup_transform_pipeline()

    def _setup_data_loader(self):
        """Set up the data loader based on configuration"""

        # Choose loader based on config
        loader_key = self.loader_config_key
        if 'h5' in loader_key.lower():
            self.loader = EnhancedH5DataLoader(self.data_config)
            logger.info(f"Using EnhancedH5DataLoader with config: {loader_key}")
        elif 'csv' in loader_key.lower() or 'nm' in loader_key.lower():
            self.loader = NMCsvDataLoader(self.data_config)
            logger.info(f"Using NMCsvDataLoader with config: {loader_key}")
        else:
            raise ValueError(f"Unsupported loader key: {loader_key}")

    def _setup_transform_pipeline(self):
        """Set up the transformation pipeline based on configuration to ensure correct order of operations"""
        # Create a pipeline
        self.pipeline = TransformPipeline(name=f"Pipeline_{self.config_hash}")

        # Get the column mapping from the loader for resolving column names
        column_mapping = self.loader.column_mapping

        # Get preprocessing configs
        preproc_config = self.config.get('preprocessing', {})

        # 1. First, extract relevant feature and output columns
        feature_cols = self.data_config.get('feature_columns', [])
        output_cols = self.data_config.get('output_columns', [])

        if feature_cols and output_cols:
            all_cols = feature_cols + output_cols
            # Ensure all columns exist in column_mapping
            valid_cols = [col for col in all_cols if col in column_mapping]

            if valid_cols:
                self.pipeline.add_transform(
                    ColumnSelector(columns=valid_cols, name="ColumnSelector")
                )
                logger.info(f"Added column selection transform for columns: {valid_cols}")

        # 2. Apply data masking for x and y coordinates
        masking_config = preproc_config.get('masking', {})
        if masking_config.get('enabled', False):
            x_range = masking_config.get('x_range', [406, 408])
            y_range = masking_config.get('y_range', [-16, 16])

            x_col = 'x' if 'x' in column_mapping else 'x_tof'
            y_col = 'y' if 'y' in column_mapping else 'y_tof'

            self.pipeline.add_transform(
                FilterRows(
                    filters={x_col: (x_range[0], x_range[1])},
                    name="X-Coordinate-Mask"
                )
            )
            logger.info(f"Added X coordinate mask: {x_range}")

            self.pipeline.add_transform(
                FilterRows(
                    filters={y_col: (y_range[0], y_range[1])},
                    name="Y-Coordinate-Mask"
                )
            )
            logger.info(f"Added Y coordinate mask: {y_range}")

        # 3. Apply pass energy transformation (You'll need to implement this transform)
        # This would require a custom transform that's not shown in your provided code
        pass_energy_config = preproc_config.get('pass_energy', {})
        if pass_energy_config.get('enabled', False) or self.data_config.get('pass_energy', False):
            # Add a placeholder for the pass energy transform
            # You would need to implement a custom PassEnergyTransform class
            logger.info("Pass energy transformation would be applied here (custom transform needed)")
            # self.pipeline.add_transform(
            #     PassEnergyTransform(name="PassEnergyTransform")
            # )

        # 4. Apply log transform to tof_values
        log_transform_config = preproc_config.get('log_transform', {})
        if log_transform_config.get('enabled', True):
            log_cols = log_transform_config.get('columns', ['tof_values'])
            base = log_transform_config.get('base', 2.0)

            # Ensure columns exist in column_mapping
            valid_log_cols = [col for col in log_cols if col in column_mapping]

            if valid_log_cols:
                self.pipeline.add_transform(
                    LogTransform(columns=valid_log_cols, base=base, name="LogTransform")
                )
                logger.info(f"Added log transform for columns: {valid_log_cols}")

        # 5. Apply feature scaling
        scaling_config = preproc_config.get('scaling', {})
        if scaling_config.get('enabled', True):
            scale_type = scaling_config.get('type', 'MinMaxScaler')
            scale_cols = scaling_config.get('feature_columns', feature_cols)

            # Ensure columns exist in column_mapping
            valid_scale_cols = [col for col in scale_cols if col in column_mapping]

            if valid_scale_cols:
                if scale_type == 'StandardScaler':
                    self.pipeline.add_transform(
                        StandardScaler(columns=valid_scale_cols, name="StandardScaler")
                    )
                    logger.info(f"Added StandardScaler for columns: {valid_scale_cols}")
                elif scale_type == 'MinMaxScaler':
                    self.pipeline.add_transform(
                        MinMaxScaler(columns=valid_scale_cols, name="MinMaxScaler")
                    )
                    logger.info(f"Added MinMaxScaler for columns: {valid_scale_cols}")

        # 6. Add feature engineering transforms
        feature_eng_config = preproc_config.get('feature_engineering', {})

        # 6.1 Add interaction terms
        interactions_config = feature_eng_config.get('interactions', {})
        if interactions_config.get('enabled', False):
            interaction_cols = interactions_config.get('columns', feature_cols)

            # Ensure columns exist in column_mapping
            valid_interaction_cols = [col for col in interaction_cols if col in column_mapping]

            if valid_interaction_cols:
                self.pipeline.add_transform(
                    PolynomialFeatures(
                        columns=valid_interaction_cols,
                        degree=2,
                        interaction_only=True,
                        name="InteractionFeatures"
                    )
                )
                logger.info(f"Added interaction features for columns: {valid_interaction_cols}")

        # 6.2 Add squared terms
        squared_config = feature_eng_config.get('squared', {})
        if squared_config.get('enabled', False):
            squared_cols = squared_config.get('columns', feature_cols)

            # Ensure columns exist in column_mapping
            valid_squared_cols = [col for col in squared_cols if col in column_mapping]

            if valid_squared_cols:
                self.pipeline.add_transform(
                    PolynomialFeatures(
                        columns=valid_squared_cols,
                        degree=2,
                        interaction_only=False,
                        name="SquaredFeatures"
                    )
                )
                logger.info(f"Added squared features for columns: {valid_squared_cols}")

        # 7. Limit number of samples if specified
        n_samples = self.data_config.get('n_samples')
        if n_samples:
            self.pipeline.add_transform(
                RandomSampler(n_samples=n_samples, name="RandomSampling")
            )
            logger.info(f"Added random sampling transform: {n_samples} samples")

        logger.info(f"Transformation pipeline setup with {len(self.pipeline.transforms)} transforms")

        # Display the order of transforms for debugging
        logger.info("Transform order:")
        for i, transform in enumerate(self.pipeline.transforms):
            logger.info(f"  {i + 1}. {transform.name}")

    def process_in_batches(self, save_separate_groups: bool = True):
        """
        Process the data in batches, applying transformations and writing to H5 files.

        Args:
            save_separate_groups: Whether to save positive and negative retardation in separate groups

        Returns:
            List of output file paths
        """
        # Load the data
        logger.info("Loading raw data...")
        raw_data = self.loader._load_raw_data()

        if raw_data is None or raw_data.size == 0:
            logger.error("No data was loaded. Check your configuration.")
            return []

        logger.info(f"Raw data loaded with shape: {raw_data.shape}")

        # Analyze the data to determine what varies
        variation_key = DatasetVariationKey.analyze_data_variations(
            raw_data, self.loader.column_mapping
        )

        # Generate a descriptive file name
        variation_desc = DatasetVariationKey.key_to_string(variation_key)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"tof_data_{variation_key}_{self.config_hash}_{timestamp}.h5"
        output_path = os.path.join(self.output_dir, output_filename)

        # Calculate total number of batches
        total_samples = raw_data.shape[0]
        total_batches = (total_samples + self.batch_size - 1) // self.batch_size

        if self.max_batches:
            total_batches = min(total_batches, self.max_batches)

        logger.info(f"Processing {total_samples} samples in {total_batches} batches")
        logger.info(f"Data variation key: {variation_key} ({variation_desc})")

        # Process data in batches
        if self.num_workers > 1:
            self._process_batches_parallel(raw_data, output_path, variation_key, total_batches, save_separate_groups)
        else:
            self._process_batches_sequential(raw_data, output_path, variation_key, total_batches, save_separate_groups)

        logger.info(f"Data processing complete. Output saved to: {output_path}")

        # Save processing summary
        self._save_processing_summary(output_path, variation_key, variation_desc)

        return [output_path]

    def _process_batches_sequential(self, raw_data: np.ndarray, output_path: str,
                                    variation_key: int, total_batches: int,
                                    save_separate_groups: bool):
        """Process batches sequentially"""
        # Fit the transformation pipeline on a sample of the data
        logger.info("Fitting transformation pipeline...")
        sample_size = min(10000, raw_data.shape[0])
        sample_indices = np.random.choice(raw_data.shape[0], sample_size, replace=False)

        # Pass column_mapping from loader to pipeline fit method
        self.pipeline.fit(raw_data[sample_indices], column_mapping=self.loader.column_mapping)

        # Initialize stats accumulators
        total_processed = 0
        total_pos_retardation = 0
        total_neg_retardation = 0

        # Open the output file
        with h5py.File(output_path, 'w') as f:
            # Save metadata
            f.attrs['creation_timestamp'] = datetime.now().isoformat()
            f.attrs['config_hash'] = self.config_hash
            f.attrs['variation_key'] = variation_key
            f.attrs['variation_description'] = DatasetVariationKey.key_to_string(variation_key)
            f.attrs['pipeline_metadata'] = json_dumps(self.pipeline.serialize())

            # Create groups for positive and negative retardation if needed
            pos_group = f.create_group('pos_retardation') if save_separate_groups else f
            neg_group = f.create_group('neg_retardation') if save_separate_groups else None

            # Find the retardation column index
            ret_idx = self.loader.column_mapping.get('retardation')

            for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
                # Get the batch
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, raw_data.shape[0])
                batch_data = raw_data[start_idx:end_idx]

                # Split batch data by retardation sign before applying transformations
                if save_separate_groups and ret_idx is not None and ret_idx < batch_data.shape[1]:
                    # Find indices for positive and negative retardation
                    pos_indices = batch_data[:, ret_idx] >= 0
                    neg_indices = ~pos_indices

                    # Process positive retardation data
                    if np.any(pos_indices):
                        pos_batch = batch_data[pos_indices]
                        pos_transformed = self.pipeline.transform(pos_batch)
                        if pos_transformed.size > 0:
                            self._append_to_dataset(pos_group, 'data', pos_transformed, batch_idx)
                            total_pos_retardation += pos_transformed.shape[0]

                    # Process negative retardation data
                    if np.any(neg_indices) and neg_group is not None:
                        neg_batch = batch_data[neg_indices]
                        neg_transformed = self.pipeline.transform(neg_batch)
                        if neg_transformed.size > 0:
                            self._append_to_dataset(neg_group, 'data', neg_transformed, batch_idx)
                            total_neg_retardation += neg_transformed.shape[0]
                else:
                    # Process all data together
                    transformed_data = self.pipeline.transform(batch_data)

                    if transformed_data.size > 0:
                        if save_separate_groups:
                            self._append_to_dataset(pos_group, 'data', transformed_data, batch_idx)
                            total_pos_retardation += transformed_data.shape[0]
                        else:
                            self._append_to_dataset(f, 'data', transformed_data, batch_idx)

                total_processed += batch_data.shape[0]

            # Process is complete, let's get a sample of transformed data for column info
            transformed_sample = None
            if 'data' in pos_group:
                transformed_sample = pos_group['data'][0:1]
            elif neg_group is not None and 'data' in neg_group:
                transformed_sample = neg_group['data'][0:1]
            elif 'data' in f:
                transformed_sample = f['data'][0:1]

            # Add column mapping information
            column_info = {
                'names': [],
                'indices': []
            }

            # Handle possible case where no data was transformed
            if transformed_sample is not None:
                for name, idx in self.loader.column_mapping.items():
                    if idx < transformed_sample.shape[1]:
                        column_info['names'].append(name)
                        column_info['indices'].append(int(idx))

                f.attrs['column_info'] = json_dumps(column_info)

            # Save statistics
            f.attrs['total_samples'] = total_processed
            f.attrs['pos_retardation_samples'] = total_pos_retardation
            f.attrs['neg_retardation_samples'] = total_neg_retardation

            # Store these in our object as well
            self.data_stats = {
                'total_samples': total_processed,
                'pos_retardation_samples': total_pos_retardation,
                'neg_retardation_samples': total_neg_retardation
            }

    def _process_batches_parallel(self, raw_data: np.ndarray, output_path: str,
                                  variation_key: int, total_batches: int,
                                  save_separate_groups: bool):
        """Process batches using multiple worker processes"""
        # Implementation of parallel processing would go here
        # This is more complex and requires careful handling of shared memory
        # For now, defer to the sequential implementation
        logger.warning("Parallel processing not fully implemented yet. Using sequential processing.")
        self._process_batches_sequential(raw_data, output_path, variation_key, total_batches, save_separate_groups)

    def _append_to_dataset(self, h5_group, dataset_name: str, data: np.ndarray, batch_idx: int):
        """Append batch data to an H5 dataset, creating it if it doesn't exist"""
        if dataset_name not in h5_group:
            # Create a resizable dataset
            maxshape = list(data.shape)
            maxshape[0] = None  # Make first dimension resizable

            h5_group.create_dataset(
                dataset_name,
                data=data,
                maxshape=maxshape,
                compression='gzip',
                compression_opts=4
            )
        else:
            # Resize and append data
            dataset = h5_group[dataset_name]
            old_size = dataset.shape[0]
            new_size = old_size + data.shape[0]
            dataset.resize(new_size, axis=0)
            dataset[old_size:new_size] = data

    def _save_processing_summary(self, output_path: str, variation_key: int, variation_desc: str):
        """Save a detailed processing summary to a separate file"""
        summary_file = os.path.splitext(output_path)[0] + '_summary.json'

        summary = {
            'output_file': os.path.basename(output_path),
            'creation_timestamp': datetime.now().isoformat(),
            'config_hash': self.config_hash,
            'variation_key': variation_key,
            'variation_description': variation_desc,
            'data_loader': self.loader.__class__.__name__,
            'loader_config_key': self.loader_config_key,
            'pipeline': {
                'name': self.pipeline.name,
                'transforms': [t.name for t in self.pipeline.transforms],
                'creation_timestamp': self.pipeline.creation_timestamp,
                'last_modified_timestamp': self.pipeline.last_modified_timestamp
            },
            'statistics': self.data_stats,
            'transform_history': self.pipeline.history
        }

        with open(summary_file, 'w') as f:
            json_dump(summary, f, indent=2)

        logger.info(f"Processing summary saved to: {summary_file}")


def process_dataset_from_config(config_path: str, num_workers: int = 1,
                                batch_size: int = 10000, max_batches: Optional[int] = None) -> List[str]:
    """
    Process a dataset based on a configuration file.

    Args:
        config_path: Path to the YAML configuration file
        num_workers: Number of worker processes
        batch_size: Batch size for processing
        max_batches: Maximum number of batches to process

    Returns:
        List of output file paths
    """
    # Load the configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create and run the processor
    processor = BatchDataProcessor(config, num_workers, batch_size, max_batches)
    output_files = processor.process_in_batches()

    return output_files


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process TOF data in batches")
    parser.add_argument('config', type=str, help='Path to the configuration YAML file')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for processing')
    parser.add_argument('--max-batches', type=int, default=None, help='Maximum number of batches to process')

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Process the dataset
    output_files = process_dataset_from_config(
        args.config, args.workers, args.batch_size, args.max_batches
    )

    print(f"Processing complete. Output files:")
    for file_path in output_files:
        print(f"  - {file_path}")