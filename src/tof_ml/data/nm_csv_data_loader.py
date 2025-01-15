import os
import csv
import glob
import logging
import numpy as np
from typing import Tuple
from src.tof_ml.data.base_data_loader import BaseDataLoader

logger = logging.getLogger(__name__)

class NMCsvDataLoader(BaseDataLoader):
    """
    DataLoader for NM-style CSV files, e.g. 'NM_neg_2.csv'.
    Produces (N,8) arrays with columns:
    [initial_ke, initial_elevation, x_tof, y_tof, mid1_ratio, mid2_ratio, retardation, tof_values].
    """

    def _parse_retardation_from_filename(self, filename: str) -> float:
        """
        Example filename: 'NM_neg_2.csv' => retardation = -2
        """
        base = os.path.basename(filename)
        name, _ = os.path.splitext(base)
        parts = name.split('_')   # ["NM", "neg", "2"]
        sign_str = parts[1]       # "neg" or "pos"
        value_str = parts[2]      # "2"
        sign = -1 if sign_str == 'neg' else 1
        return sign * float(value_str)

    def _parse_ratios_from_filename(self, filename: str) -> Tuple[float, float]:
        """
        For these NM CSVs, we pretend mid1_ratio=0.11248, mid2_ratio=0.1354.
        If you want to parse them from the filename, do so here.
        """
        return (0.11248, 0.1354)

    def _extract_pairs_from_csv(self, csv_filename: str) -> np.ndarray:
        """
        Reads the CSV in pairs of lines: the first line of each pair is 'initial conditions',
        the second is 'final conditions'. Returns an array of shape (num_pairs, 8).
        """
        ret = self._parse_retardation_from_filename(csv_filename)
        mid1, mid2 = self._parse_ratios_from_filename(csv_filename)

        data_rows = []
        with open(csv_filename, 'r', newline='') as f:
            reader = list(csv.reader(f))

            # We'll loop in steps of 2
            for i in range(0, len(reader), 2):
                if i + 1 >= len(reader):
                    break  # odd number of lines => skip last

                initial_line = reader[i]   # e.g. [..., initial_KE]
                final_line   = reader[i+1] # e.g. [..., final_TOF, final_X, ...]

                # build the Nx8 row
                # 0 => initial_ke
                # 1 => initial_elevation (hard-coded 0.0)
                # 2 => x_tof => final_line[2]
                # 3 => y_tof => 0.0
                # 4 => mid1_ratio => mid1
                # 5 => mid2_ratio => mid2
                # 6 => retardation => ret
                # 7 => tof_values => final_line[1]
                try:
                    initial_ke = float(initial_line[-1])
                    initial_elev = 0.0
                    x_val = float(final_line[2])
                    y_val = 0.0
                    tof_val = float(final_line[1])

                    row = [
                        initial_ke,
                        initial_elev,
                        mid1,
                        mid2,
                        ret,
                        tof_val,
                        x_val,
                        y_val
                    ]
                    data_rows.append(row)
                except ValueError as e:
                    logger.warning(f"Could not parse line pair in {csv_filename}: {e}")

        if not data_rows:
            return np.array([])

        return np.array(data_rows)

    def load_data(self) -> np.ndarray:
        folder_path = self.config.get('directory')
        if not folder_path:
            raise ValueError("NMCsvDataLoader requires 'folder_path' in config.")

        all_arrays = []
        for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
            arr = self._extract_pairs_from_csv(csv_file)
            if arr.size > 0:
                all_arrays.append(arr)

        if not all_arrays:
            logger.warning("No valid NM csv data found.")
            return np.array([])

        return np.vstack(all_arrays)  # shape (N,8)

    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split
        # For example:
        X = data[:, [0,2,3,4,5,7]]  # e.g., use [initial_ke, x_tof, y_tof, mid1, mid2, tof_values]
        y = data[:, 6]             # e.g., treat retardation as the 'target'
        return train_test_split(X, y, test_size=0.2, random_state=42)

