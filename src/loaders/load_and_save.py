import xarray as xr
import os
import h5py
import numpy as np


class DataGenerator:
    def __init__(self, data, scalers, batch_size=100):
        self.data = data
        self.scalers = scalers
        self.batch_size = batch_size

    def __call__(self):
        total_samples = self.data.shape[0]
        i = 0
        while i < total_samples:
            batch = self.data[i:i + self.batch_size, :]
            input_batch = self.calculate_interactions(batch[:, :5])

            output_batch = batch[:, 7:8]  # mask

            # Apply scaling -- this is done to add zeros for columns that don't exist and then remove them
            input_batch = self.scale_input(np.column_stack([input_batch, np.zeros_like(input_batch[:, :2])]))[:, :-2]

            # Replace NaNs with the log of a small value (1e-10)
            input_batch = np.nan_to_num(input_batch, nan=np.log(1e-10))

            if len(input_batch) > 0:  # Only yield if there are valid rows remaining
                yield input_batch, output_batch  # Split into inputs and outputs

            i += self.batch_size

    @staticmethod
    def calculate_interactions(input_batch):
        # Interaction terms and higher-order polynomial terms
        interaction_terms = np.column_stack([
            input_batch,
            input_batch[:, 0] * input_batch[:, 1],  # pass energy * elevation
            input_batch[:, 0] * input_batch[:, 3],  # pass energy * mid1
            input_batch[:, 0] * input_batch[:, 4],  # pass energy * mid2
            input_batch[:, 1] * input_batch[:, 2],  # elevation * retardation
            input_batch[:, 1] * input_batch[:, 3],  # elevation * mid1
            input_batch[:, 1] * input_batch[:, 4],  # elevation * mid2
            input_batch[:, 2] * input_batch[:, 3],  # retardation * mid1
            input_batch[:, 2] * input_batch[:, 4],  # retardation * mid2
            input_batch[:, 0] ** 2,                 # pass energy ^ 2
            input_batch[:, 1] ** 2,                 # elevation ^ 2
            input_batch[:, 2] ** 2,                 # retardation ^ 2
            input_batch[:, 3] ** 2,                 # mid1 ^ 2
            input_batch[:, 4] ** 2                  # mid2 ^ 2
        ])
        return interaction_terms

    def scale_input(self, scale_batch):
        scale_batch = self.scalers.transform(scale_batch)
        return scale_batch

    def inverse_scale_output(self, predictions):
        predictions = self.scalers.inverse_trainsform(predictions)
        return predictions


class DataGeneratorWithVeto(DataGenerator):
    def __init__(self, data, scalers, veto_model, batch_size=100):
        super().__init__(data, scalers, batch_size)
        self.veto_model = veto_model

    def __call__(self):
        total_samples = self.data.shape[0]
        i = 0
        while i < total_samples:
            batch = self.data[i:i + self.batch_size, :]
            input_batch = self.calculate_interactions(batch[:, :5])

            output_batch = batch[:, 5:7]  # time_of_flight and y_tof

            # Apply scaling
            full_batch = self.scale_input(np.column_stack([input_batch, output_batch]))

            # Replace NaNs with the log of a small value (1e-10)
            input_batch = np.nan_to_num(full_batch[:, :-2], nan=np.log(1e-10))
            output_batch = np.nan_to_num(full_batch[:, -2:], nan=np.log(1e-10))

            # Generate the mask using the veto model
            mask = self.veto_model.predict(input_batch, verbose=0) > 0.5
            mask = mask.flatten().astype(bool)

            # Add mask to the output batch
            output_batch = np.column_stack([output_batch, mask])

            if len(input_batch) > 0:  # Only yield if there are valid rows remaining
                yield input_batch, output_batch  # Split into inputs and outputs

            i += self.batch_size


def save_to_h5(array, filename):
    """
    Save a NumPy array to an HDF5 file.

    Parameters:
    array (np.ndarray): The NumPy array to save.
    filename (str): The name of the HDF5 file to save the array in.
    """
    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset('data', data=array)
    print(f"Data saved to {filename}")


def load_from_h5(filename):
    """
    Load a NumPy array from an HDF5 file.

    Parameters:
    filename (str): The name of the HDF5 file to load the array from.

    Returns:
    np.ndarray: The loaded NumPy array.
    """
    with h5py.File(filename, 'r') as h5f:
        array = h5f['data1'][:]
    print(f"Data loaded from {filename}")
    return array


def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data = {}
        for key in f['data1'].keys():
            data[key] = f['data1'][key][:]
    return data


def shuffle_h5(filename, out_filename):
    with h5py.File(filename, 'r') as f:
        array = f['data'][:]

    print(array.shape, array[:10])
    np.random.shuffle(array)
    print(array.shape, array[:10])

    with h5py.File(out_filename, 'w') as f:
        f.create_dataset('data', data=array)

if __name__ == '__main__':
    h5_filename = r"C:\Users\proxi\Documents\coding\TOF_ML_backup\src\simulations\combined_data.h5"
    out_filename = r"C:\Users\proxi\Documents\coding\TOF_ML_backup\src\simulations\combined_data_shuffled.h5"
    #shuffle_h5(h5_filename, out_filename)
    #data = np.random.rand(1000, 8)
    with h5py.File(h5_filename, 'r') as hf:
        data = hf["data"][:256]
    print(data)
    gen = DataGenerator(data, batch_size=256)
    for x, y in gen():
        print("Inputs:", x.shape)
        print("Outputs:", y.shape)
        break  # Only print the first batch

