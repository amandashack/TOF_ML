"""
pipeline_preprocess.py
------------------------
Refactored pipeline that:
1) Loads data from HDF5 (single or multiple),
2) Applies partial or full preprocessing steps,
3) Writes a single consolidated HDF5 with updated metadata.
"""

import os
import logging
import json
import numpy as np
import h5py
from copy import deepcopy

from src.tof_ml.utils.config_utils import load_config
from src.tof_ml.data.column_mapping import COLUMN_MAPPING
from src.tof_ml.data.h5_file_writer import H5FileWriter
from src.tof_ml.data.h5_data_loader import H5DataLoader
from src.tof_ml.data.preprocessor import DataPreprocessor
from src.tof_ml.data.data_filtering import filter_data

# --------------------------------------------------------
# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def check_existing_transformations(h5_filepath: str):
    """
    Checks the HDF5 file for attributes describing transformations
    that have already been performed. Returns a set/list of them.
    """
    if not os.path.exists(h5_filepath):
        return set()

    with h5py.File(h5_filepath, 'r') as f:
        # You could store transformations as a JSON-encoded list in an attribute
        if "transformations_applied" in f.attrs:
            return set(json.loads(f.attrs["transformations_applied"]))
        else:
            return set()


def update_transformations_metadata(f, transformations_applied, scaler_params=None):
    """
    Updates or creates top-level metadata about transformations.
    Optionally store the scaler parameters in the file as well.
    """

    # If there's an existing list in the file, merge with the new transformations
    existing_list = []
    if "transformations_applied" in f.attrs:
        existing_list = json.loads(f.attrs["transformations_applied"])

    # Merge and deduplicate
    updated_list = list(set(existing_list + transformations_applied))
    f.attrs["transformations_applied"] = json.dumps(updated_list)

    # Optionally store scaler parameters
    if scaler_params:
        # Store them in an attribute group or inline as JSON
        f.attrs["scaler_params"] = json.dumps(scaler_params)


def load_or_combine_data(config):
    """
    If config indicates multiple H5 files in a directory, combine them.
    If it's a single H5 path, read that. Return the data and the path used.
    """
    directory = config.get("directory")
    single_h5_mode = config.get("single_h5_mode", False)
    data_group_key = config.get("data_group_key", "data1")

    if single_h5_mode:
        # Load from the single file directly
        if not os.path.isfile(directory):
            raise ValueError(f"Expected a single H5 file at {directory}")
        loader = H5DataLoader(directory=None,  # Not a dir, but we can pass the path differently
                              parse_dirs=False,
                              data_group=data_group_key,
                              config=config,
                              column_mapping=COLUMN_MAPPING)
        # Modify H5DataLoader to handle single-file in its `_collect_h5_data()` or create a new SingleH5Loader
        data_array = loader.load_single_h5(directory)  # or your custom method
        return data_array, directory
    else:
        # Use existing logic that collects from multiple .h5 in a directory
        loader = H5DataLoader(directory=directory,
                              parse_dirs=config.get("parse_dirs", True),
                              data_group=data_group_key,
                              config=config,
                              column_mapping=COLUMN_MAPPING)
        data_array = loader.load_data()
        return data_array, None


def main():
    # 1) Load config
    base_config = load_config("config/base_config.yaml")
    data_config = base_config.get("data", {})

    # 2) Load or combine data
    data_array, single_h5_path = load_or_combine_data(data_config)

    # 3) Possibly filter the data (downsampling or random sampling)
    num_samples = data_config.get("n_samples")
    if num_samples is not None:
        data_array = filter_data(data_array, number_of_samples=num_samples)

    # 4) Check if scaling or log transform is needed
    #    If single h5 mode, we also check if transformations were already performed.
    transformations_performed = set()

    # If we have a single H5 we might detect existing transformations
    # If we read from multiple H5, we presumably have unscaled data
    if single_h5_path is not None:
        existing = check_existing_transformations(single_h5_path)
        transformations_performed = transformations_performed.union(existing)

    # 5) Create a preprocessor
    preprocessor_config = {
        "scaler": base_config.get("scaler", {}),
        "features": base_config.get("features", {})
    }
    preprocessor = DataPreprocessor(config=preprocessor_config, local_col_mapping=COLUMN_MAPPING)

    # 6) Apply log transform if requested and not already performed
    if preprocessor_config["scaler"].get("log_transform", False):
        if "log_transform" not in transformations_performed:
            # apply log transform
            data_array = preprocessor.log_transform(data_array)
            transformations_performed.add("log_transform")
        else:
            logger.info("Data is already log-transformed. Skipping log transform step.")

    # 7) Apply scaling if requested and not already performed
    scaling_type = preprocessor_config["scaler"].get("type", "None")
    scaler_params = None
    if scaling_type != "None":
        if "scaling" not in transformations_performed:
            data_array, scaler_params = preprocessor.apply_scaling(data_array)
            transformations_performed.add("scaling")
        else:
            logger.info("Data is already scaled. Skipping scaler step.")

    # 8) Write out to a new, single H5 file
    output_path = data_config.get("output_h5_path", "./compiled_data.h5")
    logger.info(f"Writing final preprocessed data to {output_path}")

    # Use your H5FileWriter or just directly with h5py
    with h5py.File(output_path, 'w') as f:
        # Write the final array
        f.create_dataset("processed_data", data=data_array)

        # Update transformations metadata
        update_transformations_metadata(
            f,
            transformations_applied=list(transformations_performed),
            scaler_params=scaler_params
        )

    logger.info("Done. New HDF5 file created with updated transformations.")


if __name__ == "__main__":
    main()
