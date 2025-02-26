import os
import re
import logging
import numpy as np
import h5py
from copy import deepcopy
from typing import Tuple, Optional, Dict, List
from src.tof_ml.data.base_data_loader import BaseDataLoader
from src.tof_ml.data.column_mapping import COLUMN_MAPPING

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class H5DataLoader(BaseDataLoader):
    # The patterns for parsing
    ratio_pattern = re.compile(
        r"^sim_(neg|pos)_R(\d+)_(neg|pos)_(\d+(?:\.\d+)?)_(neg|pos)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.h5$"

    )
    retardation_pattern = re.compile(r"sim_(neg|pos)_R(-?\d+)_")

    def __init__(
        self,
        directory: Optional[str] = None,
        plot_live: bool = False,
        mid1_range: Optional[List] = None,
        mid2_range: Optional[List] = None,
        retardation: Optional[List] = None,
        n_samples: Optional[int] = None,
        mask_col: Optional[str] = None,
        column_mapping: Optional[Dict] = None,
        feature_columns: Optional[List] = None,
        output_columns: Optional[List] = None,
        meta_data: Optional[Dict] = None,
        config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(
            directory=directory,
            plot_live=plot_live,
            mid1_range=mid1_range,
            mid2_range=mid2_range,
            retardation=retardation,
            n_samples=n_samples,
            mask_col=mask_col,
            column_mapping=column_mapping,
            feature_columns=feature_columns,
            output_columns=output_columns,
            meta_data=meta_data,
            config=config,
            **kwargs
        )

        self.parse_dirs = True

    def load_data(self) -> np.ndarray:
        """
        Same as before, but you may want to do:
         - read .h5
         - create (N,8) array
         - separate X,y columns
         - call self.preprocessor.fit_transform or transform
         - return final X,y or combined array
        """
        self.raw_data = self._collect_h5_data()
        self.organized_data = self._organize_data()
        # Apply pass energy conversion if requested.
        if self.pass_energy:
            self._apply_pass_energy(self.raw_data, COLUMN_MAPPING)
            self._apply_pass_energy(self.organized_data, self.column_mapping)
            self.column_mapping["pass_energy"] = self.column_mapping.pop("kinetic_energy")
        return self.organized_data


    @staticmethod
    def _apply_pass_energy(data, mapping):
        ke_idx = mapping.get("kinetic_energy")
        ret_idx = mapping.get("retardation")

        if ke_idx is not None and ret_idx is not None:
            # Convert initial_ke to pass energy (pass_energy = ke - retardation)
            # Note: This assumes that organized_data is a NumPy array.
            data[:, ke_idx] = data[:, ke_idx] - data[:, ret_idx]
        else:
            logger.warning(
                "Column mapping does not contain 'initial_ke' or 'retardation'; skipping pass energy conversion.")
        return data

    def _organize_data(self):
        data = deepcopy(self.raw_data)
        logger.info(f"Size of the dataset before masking: {data.shape}")
        if self.mask_col:
            col = self.column_mapping[self.mask_col]
            col2 = self.column_mapping['y']
            mask = ((self.raw_data[:, col] > 406) & (self.raw_data[:, col] < 408) &
                    (self.raw_data[:, col2] > -16.2) & (self.raw_data[:, col2] < 16.2))  # leave this for now
            data = self.raw_data[mask]

        if self.feature_columns and self.output_columns and self.column_mapping:
            # Convert feature names -> indices
            # e.g. if feature_cols=["retardation","tof"], then indices=[4,5]
            feature_indices = [self.column_mapping[f] for f in self.feature_columns]

            # 2) Convert output col -> index
            output_idx = [self.column_mapping[self.output_columns]]
            data = data[:, feature_indices + output_idx]

            logger.info(f"Size of the dataset after masking: {data.shape}")

            local_col_mapping = {}
            print(self.feature_columns, self.output_columns)
            for idx, col_name in enumerate(self.feature_columns + [self.output_columns]):
                local_col_mapping[col_name] = idx
            logger.info(f'Global column mapping: {self.column_mapping}')
            self.column_mapping = local_col_mapping
            logger.info(f'New column mapping: {self.column_mapping}')

        return data

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
        If mid1, mid2, or retardation is out of the allowed range, return empty array to skip.
        """
        mid1_ratio, mid2_ratio = self._parse_ratios_from_filename(filepath)
        retardation = self._parse_retardation_from_filename(filepath)

        # 1) mid1_range check
        if self.mid1_range is not None:
            low, high = self.mid1_range
            if not (low <= mid1_ratio <= high):
                logger.info(f"Skipping {filepath}; mid1={mid1_ratio} not in {self.mid1_range}")
                return np.empty((0, 8))  # shape (0,8)

        # 2) mid2_range check
        if self.mid2_range is not None:
            low, high = self.mid2_range
            if not (low <= mid2_ratio <= high):
                logger.info(f"Skipping {filepath}; mid2={mid2_ratio} not in {self.mid2_range}")
                return np.empty((0, 8))

        # 3) retardation_range check
        if self.retardation is not None:
            low, high = self.retardation
            if not (low <= retardation <= high):
                logger.info(f"Skipping {filepath}; retardation={retardation} not in {self.retardation}")
                return np.empty((0, 8))

        # If we get here, we *do* want to load the file
        with h5py.File(filepath, 'r') as f:
            initial_ke = f['data1']['initial_ke'][:]
            initial_elev = f['data1']['initial_elevation'][:]
            x_tof = f['data1']['x'][:]
            y_tof = f['data1']['y'][:]
            tof_values = f['data1']['tof'][:]

        N = len(initial_ke)
        mid1_arr = np.full((N,), mid1_ratio)
        mid2_arr = np.full((N,), mid2_ratio)
        ret_array = np.full((N,), retardation)

        data_array = np.column_stack([
            initial_ke,
            initial_elev,
            mid1_arr,
            mid2_arr,
            ret_array,
            tof_values,
            x_tof,
            y_tof,
        ])
        return data_array

    def _collect_h5_data(self) -> np.ndarray:
        if not self.directory:
            raise ValueError("H5DataLoader requires 'directory' or 'folder_path' in config.")

        all_arrays = []
        if self.parse_dirs:
            # Recurse subdirs
            subdir_pattern = re.compile(r"^R(\d+)$")
            for entry in os.listdir(self.directory):
                entry_path = os.path.join(self.directory, entry)
                if os.path.isdir(entry_path) and subdir_pattern.match(entry):
                    match = subdir_pattern.match(entry)
                    if match:
                        subdir_ret = int(match.group(1))

                        # Check if subdir_ret is in self.retardation range
                        if self.retardation is not None:
                            low, high = self.retardation
                            # If it's out of range, skip this entire subdirectory
                            if not (low <= subdir_ret <= high):
                                logger.info(
                                    f"Skipping subdir {entry_path} because {subdir_ret} not in {self.retardation}")
                                continue
                    logger.info(f"Loading data from {entry_path}")
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
        else:
            # 3) Normal behavior: load .h5 files in the given folder_path
            logger.info(f"Loading .h5 files in directory: {self.directory}")
            for fname in os.listdir(self.directory):
                if fname.endswith('.h5'):
                    base = os.path.basename(fname)
                    # Skip if ratio pattern or retardation pattern doesn't match
                    if not self.ratio_pattern.match(base) or \
                            not self.retardation_pattern.search(base):
                        logger.warning(f"Skipping file {base} - doesn't match pattern.")
                        continue

                    full_path = os.path.join(self.directory, fname)
                    arr = self._read_h5_file(full_path)
                    if arr.size > 0:
                        all_arrays.append(arr)

        if not all_arrays:
            logger.warning("No valid .h5 data found.")
            return np.array([])
        print("Returning Array.")
        return np.vstack(all_arrays)  # (N,8)

    def _update_metadata(self):
        """
        Gathers any relevant information about how the data was loaded/processed
        and stores it in self.meta_data for downstream usage (logging, DB, etc.).
        """

        self.meta_data["raw_data_size"] = self.raw_data.shape
        self.meta_data["directory"] = self.directory
        self.meta_data["mid1_range"] = self.mid1_range
        self.meta_data["mid2_range"] = self.mid2_range
        self.meta_data["retardation_range"] = self.retardation
        self.meta_data["mask_col"] = self.mask_col
        self.meta_data["samples_after_mask"] = self.organized_data.shape[0]
        self.meta_data["feature_columns"] = self.feature_columns
        self.meta_data["output_columns"] = self.output_columns
        self.meta_data["column_mapping"] = dict(self.column_mapping)
        self.meta_data["n_samples"] = self.n_samples

    def set_parse_dirs(self, parse_dirs: bool):
        self.parse_dirs = parse_dirs


if __name__ == "__main__":
    from src.tof_ml.utils.config_utils import load_config
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from src.tof_ml.data.data_filtering import filter_data
    from src.tof_ml.logging.logging_utils import setup_logger

    logger = setup_logger('trainer')

    base_config = load_config("config/base_config.yaml")
    data_config = base_config.get("data", {})
    data_loader = H5DataLoader(config=data_config, column_mapping=COLUMN_MAPPING)
    # Load data
    df = data_loader.load_data()

    data_filtered = filter_data(
        df,
        number_of_samples=10000,
        random_state=42
    )

    df_masked = pd.DataFrame(data_filtered, columns=data_loader.column_mapping.keys())

    df_raw = data_loader.raw_data
    df_raw_filtered = pd.DataFrame(filter_data(df_raw, number_of_samples=10000, random_state=42), columns=COLUMN_MAPPING.keys())

    # print statistics
    # What percentage of points remain after masking for each combination of mid1, mid2, retardation?
    # Let's start with a random point.
    def print_row(target):
        for i, column in enumerate(COLUMN_MAPPING.keys()):
            print(column, target[i])
        print('--- End of row ---\n')


    # Create a generator for the rows (assuming df_raw is indexable)
    row_gen = (df_raw[i, :].flatten() for i in range(df_raw.shape[0]))

    # Print the first 6 rows
    for _ in range(6):
        try:
            row = next(row_gen)
        except StopIteration:
            break
        print_row(row)



    # Create a scatter plot using seaborn where:
    # - x = "time_of_flight"
    # - y = "retardation"
    # - hue = "kinetic_energy" (the point color scale)
    plt.figure(figsize=(8, 6))
    scatter_plot = sns.scatterplot(
        data=df_masked,
        x="time_of_flight",
        y="retardation",
        hue="elevation",
        palette="viridis"
    )

    # Use a log (base-2) scale on both axes
    # Note: You can use plt.xscale('log', base=2) in newer matplotlib,
    # or plt.xscale('log', basex=2) for older versions.
    plt.xscale('symlog', base=2)
    plt.yscale('symlog', base=2)

    # Label the axes and show the plot
    plt.xlabel("Time of Flight (log2 scale)")
    plt.ylabel("Retardation (log2 scale)")
    plt.title("Retardation vs Time of Flight colored by Kinetic Energy")

    plt.tight_layout()
    plt.show()
