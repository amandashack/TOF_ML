import os
import glob
import csv
import numpy as np
from src.data.base_data_loader import BaseDataLoader

def parse_retardation_from_filename(filename):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split('_')
    sign_part = parts[1]
    value_part = parts[2]
    sign = -1 if sign_part == 'neg' else 1
    retardation = sign * int(value_part)
    return retardation

def extract_data_from_csv(csv_filename):
    retardation = parse_retardation_from_filename(csv_filename)
    data_rows = []
    with open(csv_filename, 'r', newline='') as f:
        reader = csv.reader(f)
        lines = list(reader)
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break
            initial_line = lines[i]
            final_line = lines[i + 1]

            initial_KE = float(initial_line[-1])
            final_TOF = float(final_line[1])
            final_X = float(final_line[2])
            data_rows.append([final_TOF, retardation, initial_KE, final_X])
    return data_rows

def build_feature_array(folder_path):
    all_data = []
    for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
        file_data = extract_data_from_csv(csv_file)
        all_data.extend(file_data)
    all_data = np.array(all_data)
    d_mask = all_data[:, -1] > 406
    data_masked = all_data[d_mask]
    return data_masked

class CSVDataLoader(BaseDataLoader):
    def load_data(self, folder_path=None):
        """
        Load data from CSV files in `folder_path`.
        Returns a NumPy array of shape (N, 4):
        [final_TOF, retardation, initial_KE, final_X]
        """
        if folder_path is None:
            raise ValueError("folder_path must be specified for CSVDataLoader.")
        data = build_feature_array(folder_path)
        return data

