import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import h5py
import glob
#from loaders.load_and_save import DataGenerator, DataGeneratorWithVeto


def load_veto_model_if_exists(checkpoint_dir, fold):
    # Define the path to the directory above the checkpoint directory
    parent_dir = os.path.abspath(os.path.join(checkpoint_dir, os.pardir))
    veto_model_pattern = os.path.join(parent_dir, "veto_model_*.h5")

    # Search for any file that matches the pattern
    veto_model_files = glob.glob(veto_model_pattern)

    if veto_model_files:
        # If there are any matching files, load the first one (you can change this behavior if needed)
        veto_model_path = veto_model_files[0]
        print(f"Loading existing veto model from {veto_model_path}")
        return tf.keras.models.load_model(veto_model_path)
    else:
        print(f"No existing veto model found in {parent_dir}. A new one will be trained.")
        return None

def calculate_scalers(data, scalers_path):
    if os.path.exists(scalers_path):
        # Load scalers from file
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        print(f"Scalers loaded from {scalers_path}")
    else:
        # Calculate interaction terms for the entire data
        #generator = DataGenerator(data, None)
        #data_with_interactions = generator.calculate_interactions(data[:, :5])
        all_data = np.column_stack([data, data[:, 5:7]])
        scalers = MinMaxScaler()
        scalers.fit(all_data)

        # Save scalers
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to {scalers_path}")

    return scalers


def partition_data(data, train_size=0.8):
    train_end = int(train_size * len(data))

    np.random.shuffle(data)
    train_data = data[:train_end]
    test_data = data[train_end:]

    return train_data, test_data


def save_test_data(test_data, out_path):
    # Check if the base directory exists, create it if it does not
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Created directory {out_path}")

    # Save test data to the specified path
    test_data_path = os.path.join(out_path, 'test_data.h5')
    with h5py.File(test_data_path, 'w') as hf:
        hf.create_dataset('test_data', data=test_data)
    print(f"Test data saved to {test_data_path}")
