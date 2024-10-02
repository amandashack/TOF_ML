import h5py
import os
import numpy as np


def parse_filename(filename):
    """
    Extract retardation, mid1_ratio, and mid2_ratio from the filename.

    Parameters:
    filename (str): The filename to parse.

    Returns:
    tuple: A tuple containing retardation, mid1_ratio, and mid2_ratio values.
    """
    # Example filename: sim_pos_R0_pos_0_pos_0_0.3.h5
    # Split filename and extract relevant parts
    parts = filename.split('_')
    if len(parts) < 8:
        return None, None, None

    # Extract retardation value
    retardation_sign = 1 if parts[1] == 'pos' else -1
    retardation = retardation_sign * int(float(parts[2][1:]))  # Remove the 'R' and convert to float

    # Extract mid1_ratio
    mid1_sign = 1 if parts[3] == 'pos' else -1
    mid1_ratio = mid1_sign * float(parts[4])

    # Extract mid2_ratio
    mid2_sign = 1 if parts[5] == 'pos' else -1
    mid2_ratio = mid2_sign * float(parts[6])

    return retardation, mid1_ratio, mid2_ratio


def load_from_h5(filename):
    """
    Load a NumPy array from an HDF5 file and include filename-derived values.

    Parameters:
    filename (str): The name of the HDF5 file to load the array from.

    Returns:
    np.ndarray: The loaded and modified NumPy array.
    """
    with h5py.File(filename, 'r') as h5f:
        initial_ke = h5f['data1']['initial_ke'][:]
        initial_elevation = h5f['data1']['initial_elevation'][:]
        x_tof = h5f['data1']['x'][:]
        y_tof = h5f['data1']['y'][:]
        tof_values = h5f['data1']['tof'][:]

    # Convert x_tof into a binary mask (1 if greater than 403.6, else 0)
    x_tof_mask = np.where(x_tof > 403.6, 1, 0)

    # Extract the values from the filename
    retardation, mid1_ratio, mid2_ratio = parse_filename(os.path.basename(filename))

    # Create columns for retardation, mid1_ratio, and mid2_ratio with the same length as the data
    retardation_col = np.full_like(initial_ke, retardation)
    mid1_ratio_col = np.full_like(initial_ke, mid1_ratio)
    mid2_ratio_col = np.full_like(initial_ke, mid2_ratio)

    # Combine all data into a single array
    array = np.column_stack([
        initial_ke, initial_elevation, retardation_col, mid1_ratio_col,
        mid2_ratio_col, tof_values, y_tof, x_tof_mask
    ])
    return array


def load_and_shuffle_all_h5_data(base_dir, random_seed=42):
    """
    Load data from all HDF5 files in the subdirectories of the base directory.

    Parameters:
    base_dir (str): The base directory containing subdirectories with HDF5 files.

    Returns:
    np.ndarray: A concatenated NumPy array containing data from all HDF5 files.
    """
    all_data = []
    for root, _, files in os.walk(base_dir):
        print(f"Searching in directory: {root}")
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                data = load_from_h5(file_path)
                all_data.append(data)

    if all_data:
        combined_data = np.concatenate(all_data)
        # Shuffle the combined data
        np.random.seed(random_seed)
        np.random.shuffle(combined_data)
        return combined_data
    else:
        print("No HDF5 files found in the directory.")
        return None


def save_combined_data(base_dir, combined_data):
    """
    Save the combined NumPy array to an HDF5 file in the base directory.

    Parameters:
    base_dir (str): The directory where the combined file should be saved.
    combined_data (np.ndarray): The data to save.
    """
    save_path = os.path.join(base_dir, 'combined_data.h5')
    with h5py.File(save_path, 'w') as h5f:
        h5f.create_dataset('combined_data', data=combined_data)
    print(f"Combined data saved to {save_path}")


if __name__ == '__main__':
    h5_file_locations = "/sdf/home/v/vkaushik/MRCO_ML_model/data/TOF_Data"
    save_location = "/sdf/home/a/ajshack"
    aggregated_data = load_and_shuffle_all_h5_data(h5_file_locations)
    if aggregated_data is not None:
        print(f"Aggregated data shape: {aggregated_data.shape}")
        save_combined_data(h5_file_locations, aggregated_data)
