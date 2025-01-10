import os
import re
import logging
import numpy as np
import h5py
from typing import Tuple
from src.data.base_data_loader import BaseDataLoader

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

    def load_data(self) -> np.ndarray:
        """
        1) If config["h5_file"] is provided (full path), load only that file.
        2) Else if config["parse_data_directories"] == True, treat folder_path as
           a base directory containing subdirectories named R(\d+) and load .h5 from each.
        3) Otherwise, load all .h5 files directly in folder_path.
        Also skip any file whose filename doesn’t match our required regex pattern.
        Returns a stacked (N,8) array.
        """
        folder_path = self.config.get('directory')
        if not folder_path:
            raise ValueError("H5DataLoader requires 'folder_path' in config.")

        # 1) If single H5 file is specified, load it only.
        h5_file = self.config.get("h5_file", None)
        if h5_file:
            if not os.path.isfile(h5_file):
                logger.error(f"Specified h5_file {h5_file} does not exist.")
                return np.array([])
            logger.info(f"Loading single .h5 file: {h5_file}")
            single_arr = self._read_h5_file(h5_file)
            return single_arr

        # 2) If parse_data_directories is True, look for subfolders matching R(\d+).
        parse_dirs = self.config.get("parse_data_directories", True)
        all_arrays = []

        if parse_dirs:
            logger.info(f"Parsing subdirectories under {folder_path} matching pattern R(\\d+).")
            subdir_pattern = re.compile(r"^R(\d+)$")

            try:
                for entry in os.listdir(folder_path):
                    entry_path = os.path.join(folder_path, entry)
                    if os.path.isdir(entry_path) and subdir_pattern.match(entry):
                        logger.info(f"Loading data from {entry_path}")
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

        return np.vstack(all_arrays)  # (N,8)

    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split
        X = data[:, [0, 1, 2, 3, 4, 6, 7]]  # some subset of columns as features
        y = data[:, 5]
        return train_test_split(X, y, test_size=0.2, random_state=42)

