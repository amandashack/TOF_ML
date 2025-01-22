import os
import re
import logging
import numpy as np
import h5py
from typing import Tuple
from src.tof_ml.data.base_data_loader import BaseDataLoader
from src.tof_ml.data.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class H5DataLoader(BaseDataLoader):
    """
    DataLoader for .h5 files. Produces (N,8) arrays with columns:
    [initial_ke, initial_elevation, x_tof, y_tof, mid1_ratio, mid2_ratio, retardation, tof_values].
    """

    # The patterns for parsing
    ratio_pattern = re.compile(
        r"^sim_(neg|pos)_R(\d+)_(neg|pos)_(\d+(?:\.\d+)?)_(neg|pos)_(\d+(?:\.\d+)?)_(\d+)\.h5$"
    )
    retardation_pattern = re.compile(r"sim_(neg|pos)_R(-?\d+)_")

    def __init__(self, config):
        super().__init__(config)
        # Optionally create the preprocessor if your config says to do so:
        preprocess_config = config.get("preprocessing", {})
        self.preprocessor = DataPreprocessor(preprocess_config) if preprocess_config else None

    def load_data(self) -> np.ndarray:
        """
        Same as before, but you may want to do:
         - read .h5
         - create (N,8) array
         - separate X,y columns
         - call self.preprocessor.fit_transform or transform
         - return final X,y or combined array
        """
        raw_data = self._collect_h5_data()

        feature_cols = self.config["features"]["input_columns"]
        output_col = self.config["features"]["output_column"]

        # 1) Convert feature names -> indices
        # e.g. if feature_cols=["retardation","tof"], then indices=[4,5]
        feature_indices = [self.column_mapping[f] for f in feature_cols]

        # 2) Convert output col -> index
        output_idx = self.column_mapping[output_col]
        X = raw_data[:, feature_indices]
        y = raw_data[:, output_idx].reshape(-1, 1)

        local_col_mapping = {}
        for idx, col_name in enumerate(feature_cols):
            local_col_mapping[col_name] = idx
        local_col_mapping[output_col] = X.shape[1]  # last column index
        logger.info('Global column mapping: ', self.column_mapping)
        self.column_mapping = local_col_mapping
        logger.info('New column mapping: ', self.column_mapping)

        if self.preprocessor:
            X_trans, y_trans = self.preprocessor.fit_transform(X, y)

            # Optionally stack back up
            data_processed = np.column_stack([X_trans, y_trans])

            return data_processed
        else:
            return np.column_stack([X, y])

    def _parse_ratios_from_filename(self, filename: str) -> Tuple[float, float]:
        """
        Extract mid1_ratio and mid2_ratio from filenames such as:
          sim_pos_R0_pos_0.5_pos_0.5_19.h5 -> mid1=+0.5, mid2=+0.5
        If not matched, returns (0.0, 0.0).
        """
        base = os.path.basename(filename)
        match = self.ratio_pattern.match(base)
        if not match:
            logger.warning(f"Filename {base} does not match the ratio pattern.")
            return (0.0, 0.0)

        sign_map = {"neg": -1, "pos": 1}
        # match.groups() => e.g. ("pos", "0", "pos", "0.5", "pos", "0.5", "19")
        mid1_sign_str = match.group(3)
        mid1_val_str = match.group(4)
        mid2_sign_str = match.group(5)
        mid2_val_str = match.group(6)

        mid1 = sign_map[mid1_sign_str] * float(mid1_val_str)
        mid2 = sign_map[mid2_sign_str] * float(mid2_val_str)
        return (mid1, mid2)

    def _parse_retardation_from_filename(self, filename: str) -> float:
        """
        e.g. 'sim_pos_R10_neg_0.25_pos_0.30_700.h5' => R10 => +10
        If not matched, returns 0.0
        """
        base = os.path.basename(filename)
        match = self.retardation_pattern.search(base)
        if not match:
            logger.warning(f"Filename {base} does not match retardation pattern.")
            return 0.0

        sign_map = {'neg': -1, 'pos': 1}
        sign_str = match.group(1)
        val_str  = match.group(2)
        return sign_map[sign_str] * float(val_str)

    def _read_h5_file(self, filepath: str) -> np.ndarray:
        """
        Opens an .h5 file and returns (N,8).
        Columns: [initial_ke, initial_elev, x_tof, y_tof, mid1_ratio, mid2_ratio, retardation, tof_values].
        """
        mid1_ratio, mid2_ratio = self._parse_ratios_from_filename(filepath)
        retardation = self._parse_retardation_from_filename(filepath)

        with h5py.File(filepath, 'r') as f:
            initial_ke = f['data1']['initial_ke'][:]
            initial_elev = f['data1']['initial_elevation'][:]
            x_tof = f['data1']['x'][:]
            y_tof = f['data1']['y'][:]
            tof_values = f['data1']['tof'][:]

        N = len(initial_ke)
        mid1_arr  = np.full((N,), mid1_ratio)
        mid2_arr  = np.full((N,), mid2_ratio)
        ret_array = np.full((N,), retardation)

        # shape => (N, 8)
        data_array = np.column_stack([
            initial_ke,
            initial_elev,
            mid1_arr,
            mid2_arr,
            ret_array,
            tof_values,
            x_tof,
            y_tof
        ])
        return data_array

    def _collect_h5_data(self) -> np.ndarray:
        """
        1) If config["h5_file"] is provided (full path), load only that file.
        2) Else if config["parse_data_directories"] == True, treat folder_path as
           a base directory containing subdirectories named R(\d+) and load .h5 from each.
        3) Otherwise, load all .h5 files directly in folder_path.
        Also skip any file whose filename doesn’t match our required regex pattern.
        Returns a stacked (N,8) array.
        """
        data_configs = self.config.get('data')
        folder_path = data_configs.get('directory')
        if not folder_path:
            raise ValueError("H5DataLoader requires 'folder_path' in config.")

        # 1) If single H5 file is specified, load it only.
        h5_file = data_configs.get("h5_file", None)
        if h5_file:
            if not os.path.isfile(h5_file):
                logger.error(f"Specified h5_file {h5_file} does not exist.")
                return np.array([])
            logger.info(f"Loading single .h5 file: {h5_file}")
            single_arr = self._read_h5_file(h5_file)
            return single_arr

        # 2) If parse_data_directories is True, look for subfolders matching R(\d+).
        parse_dirs = data_configs.get("parse_data_directories", True)
        all_arrays = []

        if parse_dirs:
            logger.info(f"Parsing subdirectories under {folder_path} matching pattern R(\\d+).")
            subdir_pattern = re.compile(r"^R(\d+)$")

            try:
                for entry in os.listdir(folder_path):
                    entry_path = os.path.join(folder_path, entry)
                    if os.path.isdir(entry_path) and subdir_pattern.match(entry):
                        logger.info(f"Loading data from {entry_path}")
                        print(f"Loading data from {entry_path}")
                        # Load .h5 files from this subdirectory
                        for fname in os.listdir(entry_path):
                            if fname.endswith('.h5'):
                                base = os.path.basename(fname)
                                # Skip if ratio pattern or retardation pattern doesn't match
                                if not self.ratio_pattern.match(base) or \
                                        not self.retardation_pattern.search(base):
                                    logger.warning(f"Skipping file {base} - doesn't match pattern.")
                                    continue

                                full_path = os.path.join(entry_path, fname)
                                arr = self._read_h5_file(full_path)
                                if arr.size > 0:
                                    all_arrays.append(arr)
                        print("Done")
            except Exception as e:
                logger.error(f"Error while parsing subdirectories: {e}")
                return np.array([])
        else:
            # 3) Normal behavior: load .h5 files in the given folder_path
            logger.info(f"Loading .h5 files in directory: {folder_path}")
            for fname in os.listdir(folder_path):
                if fname.endswith('.h5'):
                    base = os.path.basename(fname)
                    # Skip if ratio pattern or retardation pattern doesn't match
                    if not self.ratio_pattern.match(base) or \
                            not self.retardation_pattern.search(base):
                        logger.warning(f"Skipping file {base} - doesn't match pattern.")
                        continue

                    full_path = os.path.join(folder_path, fname)
                    arr = self._read_h5_file(full_path)
                    if arr.size > 0:
                        all_arrays.append(arr)

        if not all_arrays:
            logger.warning("No valid .h5 data found.")
            return np.array([])
        print("Returning Array.")
        return np.vstack(all_arrays)  # (N,8)

