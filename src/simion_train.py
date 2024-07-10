"""
This file is a sandbox for testing/running functions
"""
from plotter import one_plot_multi_scatter, pass_versus_counts
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from model_gen import run_model
import sys
import os
import tensorflow as tf
from model_eval import evaluate
import re
from loaders.load_and_save import load_from_h5, DataGenerator
import h5py

# Check GPU availability and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def calculate_scalers(data, scalers_path):
    if os.path.exists(scalers_path):
        # Load scalers from file
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        print(f"Scalers loaded from {scalers_path}")
    else:
        # Calculate interaction terms for the entire data
        generator = DataGenerator(data, None, batch_size=len(data))
        data_with_interactions = generator.calculate_interactions(data[:, :5])

        # Define and fit scalers
        scalers = {
            'log_standard': StandardScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler()
        }
        scalers['log_standard'].fit(data_with_interactions[:, :1])  # Log pass energy
        scalers['robust'].fit(data_with_interactions[:, 1:3])  # Elevation and retardation
        scalers['maxabs'].fit(data_with_interactions[:, 3:])  # Ratios and interaction terms

        # Save scalers
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to {scalers_path}")

    return scalers


def partition_data(data, train_size=0.7, val_size=0.15):
    train_end = int(train_size * len(data))
    val_end = int(val_size * len(data)) + train_end

    np.random.shuffle(data)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


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


def run_train(out_path, params, h5_filename, checkpoint_dir):
    """
    Load data from an HDF5 file, split it into training, validation, and test sets,
    and train the model using a generator.

    Parameters:
    out_path (str): The path to save the trained model.
    params (dict): Parameters for the model.
    h5_filename (str): The name of the HDF5 file containing the data.
    checkpoint_dir (str): Directory to save model checkpoints.
    """
    # Load data from the HDF5 file
    with h5py.File(h5_filename, 'r') as hf:
        data = hf["data"][:]

    # Define the path for scalers
    scalers_path = os.path.join(out_path, 'scalers.pkl')

    # Calculate or load scalers
    scalers = calculate_scalers(data, scalers_path)

    # Split data into train, val, test sets
    train_data, val_data, test_data = partition_data(data)

    # Save test data
    save_test_data(test_data, out_path)

    # Define batch size and steps per epoch
    batch_size = int(params["batch_size"])
    params['steps_per_epoch'] = np.ceil(len(train_data) / batch_size).astype(int)
    params['validation_steps'] = np.ceil(len(val_data) / batch_size).astype(int)

    # Create generator instances for training, validation, and test datasets
    train_gen = DataGenerator(train_data, scalers, batch_size=batch_size)
    val_gen = DataGenerator(val_data, scalers, batch_size=batch_size)
    test_gen = DataGenerator(test_data, scalers, batch_size=batch_size)

    # Create tf.data.Dataset from the generators
    train_dataset = tf.data.Dataset.from_generator(
        train_gen, output_signature=(
            tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    ).take(len(train_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        val_gen, output_signature=(
            tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    ).take(len(val_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_generator(
        test_gen, output_signature=(
            tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    ).take(len(test_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

    # Train the model
    model, history = run_model(train_dataset, val_dataset, params, checkpoint_dir)

    model.save(out_path)

    # Evaluate the model
    loss_test = evaluate(model, test_dataset, batch_size=batch_size)
    print(f"test_loss {loss_test}")

if __name__ == '__main__':
    h5_filename = r"C:\Users\proxi\Documents\coding\TOF_ML_backup\src\simulations\combined_data_shuffled.h5"
    run_train("/Users/proxi/Documents/coding/stored_models/test_001/15",
              {"layer_size": 64, "batch_size": 256, 'dropout': 0.4,
               'learning_rate': 0.1, 'optimizer': 'Adam'},
              h5_filename, r"C:\Users\proxi\Documents\coding\stored_models\test_001\14\checkpoints")
