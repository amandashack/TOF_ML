import os
import re
import logging
import numpy as np
import h5py

data_loader_logger = logging.getLogger("data_loader")


def parse_filename(filename):
    pattern = re.compile(
        r"sim_(neg|pos)_R(-?\d+)_(neg|pos)_(-?\d+\.\d+)_(neg|pos)_(-?\d+\.\d+)_(\d+)\.h5"
    )
    match = pattern.match(filename)
    if not match:
        return None

    sign_map = {'neg': -1, 'pos': 1}

    retardation_sign = sign_map[match.group(1)]
    retardation_value = int(match.group(2))
    mid1_ratio_sign = sign_map[match.group(3)]
    mid1_ratio_value = float(match.group(4))
    mid2_ratio_sign = sign_map[match.group(5)]
    mid2_ratio_value = float(match.group(6))
    kinetic_energy_meta = int(match.group(7))  # Not used directly for plotting, but parsed anyway

    return {
        'retardation': retardation_sign * retardation_value,
        'mid1_ratio': mid1_ratio_sign * mid1_ratio_value,
        'mid2_ratio': mid2_ratio_sign * mid2_ratio_value,
        'kinetic_energy_meta': kinetic_energy_meta
    }


def read_h5_file(filename):
    with h5py.File(filename, 'r') as f:
        data = {
            'initial_ke': f['data1']['initial_ke'][:],
            'initial_elevation': f['data1']['initial_elevation'][:],
            'x_tof': f['data1']['x'][:],
            'y_tof': f['data1']['y'][:],
            'tof_values': f['data1']['tof'][:],
            'final_elevation': f['data1']['final_elevation'][:],
            'final_ke': f['data1']['final_ke'][:],
        }
    return data


def load_stacked_data(data_dir: str) -> np.ndarray:
    """
    Loads all H5 files in data_dir and returns a stacked numpy array of shape (N, 8).
    Columns: [initial_ke, initial_elevation, x_tof, y_tof, mid1_ratio, mid2_ratio, retardation, tof_values]
    """

    data_loader_logger.info(f"Loading data from directory: {data_dir}")
    stacked_data = []

    for fname in os.listdir(data_dir):
        if fname.endswith(".h5"):
            full_path = os.path.join(data_dir, fname)
            file_metadata = parse_filename(fname)
            if file_metadata is None:
                data_loader_logger.warning(f"Filename {fname} does not match expected pattern, skipping.")
                continue

            data_loader_logger.debug(f"Reading file: {fname}")
            file_data = read_h5_file(full_path)

            initial_ke = file_data['initial_ke']
            initial_elev = file_data['initial_elevation']
            x_tof = file_data['x_tof']
            y_tof = file_data['y_tof']
            tof_values = file_data['tof_values']

            N = len(initial_ke)
            mid1_ratio_array = np.full((N,), file_metadata['mid1_ratio'])
            mid2_ratio_array = np.full((N,), file_metadata['mid2_ratio'])
            retardation_array = np.full((N,), file_metadata['retardation'])

            file_array = np.column_stack([
                initial_ke,
                initial_elev,
                x_tof,
                y_tof,
                mid1_ratio_array,
                mid2_ratio_array,
                retardation_array,
                tof_values
            ])

            stacked_data.append(file_array)

    if not stacked_data:
        data_loader_logger.error("No valid data files found.")
        return np.array([])

    final_array = np.vstack(stacked_data)
    data_loader_logger.info(f"Data loaded successfully. Final shape: {final_array.shape}")
    return final_array
