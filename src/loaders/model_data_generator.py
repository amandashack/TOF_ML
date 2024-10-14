import xarray as xr
import os
import h5py
import numpy as np
import keras
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def create_dataset(indices, data_filepath, batch_size, num_workers, worker_index, shuffle=True):
    input_dim = 4  # Adjust based on your data
    output_dim = 1

    # Manually shard the indices
    indices = indices[worker_index::num_workers]

    # Create a dataset of indices
    indices_dataset = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        indices_dataset = indices_dataset.shuffle(buffer_size=10000)

    # Batch the indices to read data in batches
    indices_dataset = indices_dataset.batch(batch_size)

    def _parse_function(batch_indices):
        def py_function(batch_indices):
            # Open the HDF5 file once per batch
            with h5py.File(data_filepath, 'r') as hf:
                batch_indices_np = np.sort(batch_indices.numpy())
                data_rows = hf['combined_data'][batch_indices_np]

                input_list = []
                output_list = []
                for data_row in data_rows:
                    if data_row.ndim == 1:
                        data_row = np.expand_dims(data_row, axis=0)
                    mask = data_row[:, -1].astype(bool)
                    masked_data_row = data_row[mask]
                    if masked_data_row.shape[0] == 0:
                        continue
                    input_data = masked_data_row[:, [2, 3, 4, 5]]
                    output_data = masked_data_row[:, 0].reshape(-1, 1)
                    output_data = np.log2(output_data)
                    input_list.append(input_data.astype(np.float32))
                    output_list.append(output_data.astype(np.float32))
                # Concatenate the lists
                if input_list and output_list:
                    inputs = np.concatenate(input_list, axis=0)
                    outputs = np.concatenate(output_list, axis=0)
                    return inputs, outputs
                else:
                    # Return empty arrays if no data
                    return np.empty((0, input_dim), dtype=np.float32), np.empty((0, output_dim), dtype=np.float32)
        inp, outp = tf.py_function(py_function, [batch_indices], [tf.float32, tf.float32])
        inp.set_shape([None, input_dim])
        outp.set_shape([None, output_dim])
        return inp, outp

    # Map the parsing function over the batches
    dataset = indices_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.unbatch().repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Set the auto-shard policy to OFF since we are sharding manually
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    dataset = dataset.with_options(options)

    return dataset



def calculate_and_save_scalers(indices, data_filepath, scalers_path):
    if os.path.exists(scalers_path):
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
            min_values = scalers['min_values']
            max_values = scalers['max_values']
            print(f"Scalers loaded from {scalers_path}")
            return min_values, max_values
    else:
        # Read the entire dataset into memory
        with h5py.File(data_filepath, 'r') as hf:
            data = hf['combined_data'][:]

        # Filter data using indices
        data = data[indices]

        # Apply mask to filter valid data
        mask = data[:, -1].astype(bool)
        data = data[mask]

        # Apply log transforms
        data[:, 0] = np.log2(data[:, 0])  # Log transform of output
        data[:, 5] = np.log2(data[:, 5])  # Log transform of sixth feature

        # Process input data
        input_data = data[:, [2, 3, 4, 5]]  # Adjust indices based on your data

        # Apply interaction terms
        x1 = input_data[:, 0:1]
        x2 = input_data[:, 1:2]
        x3 = input_data[:, 2:3]
        x4 = input_data[:, 3:4]
        interaction_terms = np.hstack([
            x1 * x2,
            x1 * x3,
            x1 * x4,
            x2 * x3,
            x2 * x4,
            x3 * x4,
            x4 ** 2
        ])
        processed_input = np.hstack([input_data, interaction_terms])

        # Combine output and input data
        output_data = data[:, 0].reshape(-1, 1)  # Output variable
        full_data = np.hstack([output_data, processed_input])

        # Fit the scaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(full_data)
        min_values = scaler.data_min_
        max_values = scaler.data_max_

        # Save the scalers to disk
        with open(scalers_path, 'wb') as f:
            pickle.dump({'min_values': min_values, 'max_values': max_values}, f)
            print(f"Scalers saved to {scalers_path}")

        return min_values, max_values


# Function to create datasets using tf.data.Dataset
def create_dataset2(indices, data_filepath, batch_size, shuffle=True):
    # Adjust input_dim based on the new number of features after preprocessing
    input_dim = 4  # Original features + interaction terms
    output_dim = 1

    def data_generator():
        with h5py.File(data_filepath, 'r') as hf:
            for idx in indices:
                data_row = hf['combined_data'][idx]
                if data_row.ndim == 1:
                    data_row = np.expand_dims(data_row, axis=0)
                mask = data_row[:, -1].astype(bool)
                masked_data_row = data_row[mask]
                if masked_data_row.shape[0] == 0:
                    continue
                input_data = masked_data_row[:, [2, 3, 4, 5]]
                output_data = masked_data_row[:, 0].reshape(-1, 1)

                # Apply log2 transformation to specific features
                input_data[:, 3] = np.log2(input_data[:, 3])
                output_data = np.log2(output_data)

                # Note: Scaling is generally not necessary for Random Forests
                for inp, outp in zip(input_data, output_data):
                    yield inp.astype(np.float32), outp.astype(np.float32)

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([input_dim]), tf.TensorShape([output_dim]))
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.batch(batch_size).cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Remove the .repeat() method to avoid infinite loops during training
    # Adjust sharding options if necessary
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    return dataset


def load_scalers(scalers_path):
    if os.path.exists(scalers_path):
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
            min_values = scalers['min_values']
            max_values = scalers['max_values']
            print(f"Scalers loaded from {scalers_path}")
            return min_values, max_values
    else:
        raise FileNotFoundError(f"Scalers file not found at {scalers_path}")


def process_input(input_data):
    x1 = input_data[:, 0:1]
    x2 = input_data[:, 1:2]
    x3 = input_data[:, 2:3]
    x4 = input_data[:, 3:4]
    interaction_terms = np.hstack([
        x1 * x2,
        x1 * x3,
        x1 * x4,
        x2 * x3,
        x2 * x4,
        x3 * x4,
        x4 ** 2
    ])
    return np.hstack([input_data, interaction_terms])


def calculate_and_save_scalers(indices, data_filepath, scalers_path):
    if os.path.exists(scalers_path):
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
            min_values = scalers['min_values']
            max_values = scalers['max_values']
            print(f"Scalers loaded from {scalers_path}")
            return min_values, max_values
    else:
        # Read the entire dataset into memory
        with h5py.File(data_filepath, 'r') as hf:
            data = hf['combined_data'][:]

        # Filter data using indices
        data = data[indices]

        # Apply mask to filter valid data
        mask = data[:, -1].astype(bool)
        data = data[mask]

        # Apply log transforms
        data[:, 0] = np.log2(data[:, 0])  # Log transform of output
        data[:, 5] = np.log2(data[:, 5])  # Log transform of sixth feature

        # Process input data
        input_data = data[:, [2, 3, 4, 5]]  # Adjust indices based on your data

        # Apply interaction terms
        interaction_data = process_input(input_data)

        # Combine output and input data
        output_data = data[:, 0].reshape(-1, 1)  # Output variable
        full_data = np.hstack([output_data, interaction_data])

        # Fit the scaler
        scaler = MinMaxScaler()
        scaler.fit(full_data)
        min_values = scaler.data_min_
        max_values = scaler.data_max_

        # Save the scalers to disk
        with open(scalers_path, 'wb') as f:
            pickle.dump({'min_values': min_values, 'max_values': max_values}, f)
            print(f"Scalers saved to {scalers_path}")

        return min_values, max_values


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, data_filename, scalers_path,
                 batch_size=32, dim=(32, 5), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.data_filename = data_filename
        self.shuffle = shuffle
        self.scalers_path = scalers_path
        self.scalers = None
        self.on_epoch_end()

        # Calculate or load scalers

    def calculate_scalers(self):
        if self.scalers_path and os.path.exists(self.scalers_path):
            with open(self.scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
            print(f"Scalers loaded from {self.scalers_path}")
        else:
            # Calculate scalers using training data
            with h5py.File(self.data_filename, 'r') as hf:
                data = hf['combined_data'][:]
                data[:, 0] = np.log2(data[:, 0])
                data[:, 5] = np.log2(data[:, 5])
                interaction_data = self.process_input(data[:, :5])  # Assume using first 5 columns for interactions
                all_data = np.column_stack([interaction_data, data[:, 5:7]])
                self.scalers = MinMaxScaler().fit(all_data)

                if self.scalers_path:
                    with open(self.scalers_path, 'wb') as f:
                        pickle.dump(self.scalers, f)
                    print(f"Scalers saved to {self.scalers_path}")

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self._data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, 2))  # Expecting 2 output columns: time_of_flight and y_tof

        with h5py.File(self.data_filename, 'r') as hf:
            for i, ID in enumerate(list_IDs_temp):
                data_row = hf['combined_data'][ID]

                # Ensure data_row is at least 2-dimensional
                if data_row.ndim == 1:
                    data_row = np.expand_dims(data_row, axis=0)

                data_row[:, 0] = np.log2(data_row[:, 0])
                data_row[:, 5] = np.log2(data_row[:, 5])
                input_data = self.process_input(data_row[:, :5])

                # Apply scaling
                input_data = self.scalers.transform(
                    np.column_stack([input_data, np.zeros_like(input_data[:, :2])])
                )[:, :-2]
                input_data = np.nan_to_num(input_data, nan=np.log(1e-10))  # Replace NaNs

                X[i,] = input_data
                y[i,] = data_row[:, 5:7]  # Extract columns 6 and 7 (time_of_flight and y_tof)

        return X, y

    @staticmethod
    def process_input(input_data):
        'Applies interaction terms and preprocessing to input data'
        interaction_terms = np.column_stack([
            input_data,
            input_data[:, 0] * input_data[:, 1],  # pass energy * elevation
            input_data[:, 0] * input_data[:, 3],  # pass energy * mid1
            input_data[:, 0] * input_data[:, 4],  # pass energy * mid2
            input_data[:, 1] * input_data[:, 2],  # elevation * retardation
            input_data[:, 1] * input_data[:, 3],  # elevation * mid1
            input_data[:, 1] * input_data[:, 4],  # elevation * mid2
            input_data[:, 2] * input_data[:, 3],  # retardation * mid1
            input_data[:, 2] * input_data[:, 4],  # retardation * mid2
            input_data[:, 0] ** 2,  # pass energy ^ 2
            input_data[:, 1] ** 2,  # elevation ^ 2
            input_data[:, 2] ** 2,  # retardation ^ 2
            input_data[:, 3] ** 2,  # mid1 ^ 2
            input_data[:, 4] ** 2  # mid2 ^ 2
        ])
        return interaction_terms


class DataGeneratorWithVeto(DataGenerator):
    def __init__(self, list_IDs, labels, data_filename, veto_model,
                 batch_size=32, dim=(18,), shuffle=True, scalers_path=None):
        super().__init__(list_IDs, labels, data_filename, scalers_path,
                         batch_size=batch_size, dim=dim, shuffle=shuffle)
        self.veto_model = veto_model

    def _data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, 2))  # Assuming 2 features for output (time_of_flight and y_tof)

        with h5py.File(self.data_filename, 'r') as hf:
            for i, ID in enumerate(list_IDs_temp):
                data_row = hf['combined_data'][ID]

                # Ensure data_row is at least 2-dimensional
                if data_row.ndim == 1:
                    data_row = np.expand_dims(data_row, axis=0)

                data_row[:, 0] = np.log2(data_row[:, 0])
                data_row[:, 5] = np.log2(data_row[:, 5])
                input_data = self.process_input(data_row[:, :5])
                output_data = data_row[:, 5:7]  # time_of_flight and y_tof

                # Apply scaling
                full_batch = self.scalers.transform(np.column_stack([input_data, np.zeros_like(input_data[:, :2])]))[:, :-2]
                input_data = np.nan_to_num(full_batch, nan=np.log(1e-10))

                # Generate mask using the veto model
                mask = self.veto_model.predict(input_data[np.newaxis, ...], verbose=0) > 0.5
                if mask:
                    X[i,] = input_data
                    y[i,] = output_data

        return X, y


class DataGeneratorTofToKE(DataGenerator):
    def __init__(self, list_IDs, labels, data_filename, batch_size=32,
                 dim=(18,), shuffle=True, scalers_path=None):
        super().__init__(list_IDs, labels, data_filename, scalers_path,
                         batch_size=batch_size, dim=dim, shuffle=shuffle)

    def _data_generation(self, list_IDs_temp):
        input_list = []
        output_list = []
        samples_collected = 0
        max_samples = self.batch_size

        with h5py.File(self.data_filename, 'r') as hf:
            idx = 0  # Index within list_IDs_temp
            while samples_collected < max_samples and idx < len(list_IDs_temp):
                ID = list_IDs_temp[idx]
                data_row = hf['combined_data'][ID]

                # Ensure data_row is at least 2-dimensional
                if data_row.ndim == 1:
                    data_row = np.expand_dims(data_row, axis=0)
                epsilon = 1e-10  # To avoid log(0)
                data_row[:, 0] = np.log2(data_row[:, 0] + epsilon)
                data_row[:, 5] = np.log2(data_row[:, 5] + epsilon)
                mask = data_row[:, -1].astype(bool)
                masked_data_row = data_row[mask]

                if masked_data_row.shape[0] == 0:
                    idx += 1
                    continue  # Skip data points with no valid samples

                # Extract input and output data
                input_data = masked_data_row[:, [2, 3, 4, 5]]
                output_data = masked_data_row[:, 0].reshape(-1, 1)

                # **Add this line to process input_data**
                input_data = self.process_input(input_data)

                # Stack output and input data
                full_batch = np.hstack([output_data, input_data])

                # Apply scaling
                scaled_full_batch = self.scalers.transform(full_batch)
                scaled_output_data = scaled_full_batch[:, 0]
                scaled_input_data = scaled_full_batch[:, 1:]

                # Collect the scaled data
                for inp, outp in zip(scaled_input_data, scaled_output_data):
                    input_list.append(inp)
                    output_list.append(outp)
                    samples_collected += 1
                    if samples_collected >= max_samples:
                        break

                idx += 1  # Move to the next ID

        # Convert lists to arrays
        X = np.array(input_list)
        y = np.array(output_list).reshape(-1, 1)

        return X, y

    def calculate_scalers(self):
        if self.scalers_path and os.path.exists(self.scalers_path):
            with open(self.scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
            print(f"Scalers loaded from {self.scalers_path}")
        else:
            # Calculate scalers using training data
            with h5py.File(self.data_filename, 'r') as hf:
                data = hf['combined_data'][:]
                epsilon = 1e-10
                data[:, 0] = np.log2(data[:, 0] + epsilon)
                data[:, 5] = np.log2(data[:, 5] + epsilon)
                mask = data[:, -1].astype(bool)
                data = data[mask]

                # Extract the specific columns you need
                output_data = data[:, 0].reshape(-1, 1)  # Energy
                input_data = data[:, [2, 3, 4, 5]]
                input_data = self.process_input(input_data)  # Process input data

                # Stack output and input data
                full_data = np.hstack([output_data, input_data])

                # Fit the scaler on the relevant columns
                self.scalers = MinMaxScaler().fit(full_data)

                if self.scalers_path:
                    with open(self.scalers_path, 'wb') as f:
                        pickle.dump(self.scalers, f)
                    print(f"Scalers saved to {self.scalers_path}")

    @staticmethod
    def process_input(input_data):
        'Applies interaction terms and preprocessing to input data'
        interaction_terms = np.column_stack([
            input_data,
            input_data[:, 0] * input_data[:, 1],
            input_data[:, 0] * input_data[:, 2],
            input_data[:, 0] * input_data[:, 3],
            input_data[:, 1] * input_data[:, 2],
            input_data[:, 1] * input_data[:, 3],
            input_data[:, 2] * input_data[:, 3],
            input_data[:, 3] ** 2,
        ])
        return interaction_terms


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


def load_from_h5(filename, grp='data1'):
    """
    Load a NumPy array from an HDF5 file.

    Parameters:
    filename (str): The name of the HDF5 file to load the array from.

    Returns:
    np.ndarray: The loaded NumPy array.
    """
    with h5py.File(filename, 'r') as h5f:
        array = h5f[grp][:]
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

