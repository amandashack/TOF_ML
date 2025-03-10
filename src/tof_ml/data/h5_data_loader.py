# src/tof_ml/data/h5_data_loader.py

import os
import re
import json
import logging
import numpy as np
import h5py
from typing import Optional, List, Dict, Tuple, Any
from src.tof_ml.data.base_data import BaseDataLoader

logger = logging.getLogger(__name__)


class H5MetadataReader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None

    def __enter__(self):
        self.file = h5py.File(self.filepath, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def get_metadata(self, group_path='/'):
        """
        Recursively retrieve metadata from HDF5.
        """
        metadata = {}
        group = self.file[group_path]
        for key, value in group.attrs.items():
            if key not in ['collection_efficiency', 'ks_score']:
                metadata[key] = value
        return metadata


class H5DataLoader(BaseDataLoader):
    """
    Reads H5 files from a directory, merges them if they are consistent,
    returns data + metadata about what's already done (pass_energy/scaling).
    """

    ratio_pattern = re.compile(
        r"^sim_(neg|pos)_R(\d+)_(neg|pos)_(\d+(?:\.\d+)?)_(neg|pos)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.h5$"
    )
    retardation_pattern = re.compile(r"sim_(neg|pos)_R(-?\d+)_")

    def __init__(self, config: Optional[Dict] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        # parse_dirs, subdir_pattern, data_group from config or kwargs
        self.parse_dirs = kwargs.get("parse_dirs", True)
        pattern = kwargs.get("subdir_pattern", r"^R(\d+)$")
        self.subdir_pattern = re.compile(pattern)
        self.data_group = kwargs.get("data_group", "data1")

        self.global_file_meta: Optional[Dict[str, Any]] = None

    def load_data(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns (final_data, loader_meta).
        - final_data: stacked data from all consistent files
        - loader_meta: the final "global_file_meta" with mid1_range, mid2_range,
          and retardation_range overwritten by actual min/max from final_data.
        """
        if not os.path.isdir(self.directory):
            raise ValueError(f"Invalid directory: {self.directory}")

        all_arrays = []
        if self.parse_dirs:
            for entry in os.listdir(self.directory):
                subpath = os.path.join(self.directory, entry)
                if os.path.isdir(subpath) and self.subdir_pattern.match(entry):
                    arrays_in_subdir = self._collect_from_dir(subpath)
                    all_arrays.extend(arrays_in_subdir)
        else:
            arrays_in_subdir = self._collect_from_dir(self.directory)
            all_arrays.extend(arrays_in_subdir)

        if not all_arrays:
            logger.warning("No valid .h5 found or all were skipped. Returning empty array.")
            return np.array([]), {}

        final_data = np.vstack(all_arrays)

        # Overwrite the range keys in global_file_meta with final actual min/max
        if final_data.size > 0 and self.column_mapping and self.global_file_meta is not None:
            self._update_file_meta_ranges(final_data)

        return final_data, self.global_file_meta if self.global_file_meta else {}

    def _collect_from_dir(self, path: str) -> List[np.ndarray]:
        arrays = []
        for fname in os.listdir(path):
            if fname.endswith(".h5"):
                full_path = os.path.join(path, fname)
                if not self._filename_matches(fname):
                    continue

                file_meta = self._parse_file_metadata(full_path)
                if not self._check_consistency(file_meta, fname):
                    continue

                # If we get here, the file is consistent => read data
                arr = self._read_data_file(full_path)
                if arr.size > 0:
                    arrays.append(arr)
        return arrays

    def _filename_matches(self, fname: str) -> bool:
        base = os.path.basename(fname)
        if not self.ratio_pattern.match(base):
            return False
        if not self.retardation_pattern.search(base):
            return False
        return True

    def _parse_file_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Return the entire metadata dict from the file,
        including recognized and unrecognized attributes.
        """
        with H5MetadataReader(filepath) as reader:
            fm = reader.get_metadata(self.data_group)  # parse everything at the root
            #print(f"Here is the metadata from file {filepath}: {fm}")
        return fm

    def _check_consistency(self, file_meta: Dict[str, Any], fname: str) -> bool:
        if self.global_file_meta is None:
            self.global_file_meta = file_meta
            return True

        if file_meta != self.global_file_meta:
            print(file_meta, self.global_file_meta)
            # DEBUG: Print out or log differences so you can see what's causing the mismatch
            logger.warning(
                f"Skipping {fname} due to mismatch in file-level metadata.\n"
                f"global_file_meta={self.global_file_meta}\n"
                f"file_meta={file_meta}"
            )
            return False
        return True

    def _deep_search(self, meta: dict, key: str):
        if key in meta:
            return meta[key]
        for k,v in meta.items():
            if isinstance(v, dict):
                r = self._deep_search(v, key)
                if r is not None:
                    return r
        return None

    def _extract_list(self, meta: dict, key: str) -> List[str]:
        val = self._deep_search(meta, key)
        if not val:
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                arr = json.loads(val)
                if isinstance(arr, list):
                    return arr
            except:
                pass
        return []

    def _read_data_file(self, filepath: str) -> np.ndarray:
        """
        1) Parse mid1, mid2, ret from the filename
        2) If out of config-supplied range, skip by returning an empty array
        3) Load data from the h5 group
        """
        mid1_val, mid2_val = self._parse_ratios(filepath)
        ret_val = self._parse_retardation(filepath)

        # domain checks
        if self.mid1_range and not (self.mid1_range[0] <= mid1_val <= self.mid1_range[1]):
            logger.info(f"Skipping {filepath}: mid1 out of range.")
            return np.empty((0,8))
        if self.mid2_range and not (self.mid2_range[0] <= mid2_val <= self.mid2_range[1]):
            logger.info(f"Skipping {filepath}: mid2 out of range.")
            return np.empty((0,8))
        if self.retardation and not (self.retardation[0] <= ret_val <= self.retardation[1]):
            logger.info(f"Skipping {filepath}: ret out of range.")
            return np.empty((0,8))

        with h5py.File(filepath, 'r') as f:
            g = f[self.data_group]
            initial_ke = g['initial_ke'][:]
            elev = g['initial_elevation'][:]
            x_tof = g['x'][:]
            y_tof = g['y'][:]
            tof_vals = g['tof'][:]

        N = len(initial_ke)
        mid1_arr = np.full((N,), mid1_val)
        mid2_arr = np.full((N,), mid2_val)
        ret_arr  = np.full((N,), ret_val)

        data_array = np.column_stack([
            initial_ke,
            elev,
            x_tof,
            y_tof,
            mid1_arr,
            mid2_arr,
            ret_arr,
            tof_vals
        ])
        return data_array

    def _parse_ratios(self, filepath: str) -> Tuple[float, float]:
        base = os.path.basename(filepath)
        m = self.ratio_pattern.match(base)
        if not m:
            return (0.0, 0.0)
        sign_map = {'neg': -1, 'pos': 1}
        # groups => (neg|pos, R(\d+), neg|pos, \d+(?:\.\d+)? ...
        mid1_sign_str = m.group(3)
        mid1_val_str  = m.group(4)
        mid2_sign_str = m.group(5)
        mid2_val_str  = m.group(6)
        mid1 = sign_map[mid1_sign_str] * float(mid1_val_str)
        mid2 = sign_map[mid2_sign_str] * float(mid2_val_str)
        return (mid1, mid2)

    def _parse_retardation(self, filepath: str) -> float:
        base = os.path.basename(filepath)
        m = self.retardation_pattern.search(base)
        if not m:
            return 0.0
        sign_map = {'neg': -1, 'pos': 1}
        sign_str = m.group(1)
        val_str  = m.group(2)
        return sign_map[sign_str]*float(val_str)

    def _update_file_meta_ranges(self, final_data: np.ndarray) -> None:
        """
        Overwrites mid1_range, mid2_range, and retardation_range in self.global_file_meta
        with the actual min/max from final_data.
        """
        def _replace_range(meta_key: str, col_name: str):
            if col_name not in self.column_mapping:
                return
            idx = self.column_mapping[col_name]
            col_min = float(final_data[:, idx].min())
            col_max = float(final_data[:, idx].max())
            self.global_file_meta[meta_key] = [col_min, col_max]

        # Overwrite
        _replace_range("mid1_range", "mid1")
        _replace_range("mid2_range", "mid2")
        _replace_range("retardation_range", "retardation")

