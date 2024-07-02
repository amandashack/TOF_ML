"""
This file is a sandbox for testing/running functions
"""
from plotter import one_plot_multi_scatter, pass_versus_counts
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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


def partition_data(h5_filename, train_size=0.7, val_size=0.15):
    with h5py.File(h5_filename, 'r') as hf:
        data = hf["data"][:]

    train_end = int(train_size * len(data))
    val_end = int(val_size * len(data)) + train_end

    np.random.shuffle(data)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def save_test_data(test_data, out_path):
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
    # Partition the data
    train_data, val_data, test_data = partition_data(h5_filename)
    print(train_data.shape, val_data.shape, test_data.shape)

    # Save test data
    save_test_data(test_data, out_path)

    # Define batch size
    batch_size = int(params["batch_size"])

    # Calculate steps per epoch
    params['steps_per_epoch'] = np.ceil(len(train_data) / batch_size).astype(int)
    params['validation_steps'] = np.ceil(len(val_data) / batch_size).astype(int)

    # Create generator instances for training, validation, and test datasets
    train_gen = DataGenerator(train_data, batch_size=batch_size)
    val_gen = DataGenerator(val_data, batch_size=batch_size)

    # Create tf.data.Dataset from the generators
    train_dataset = tf.data.Dataset.from_generator(
        train_gen, output_signature=(
            tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
        )
    ).repeat().cache().prefetch(tf.data.experimental.AUTOTUNE)  # Added repeat()

    val_dataset = tf.data.Dataset.from_generator(
        val_gen, output_signature=(
            tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
        )
    ).repeat().cache().prefetch(tf.data.experimental.AUTOTUNE)  # Added repeat()

    # Train the model
    model, history = run_model(train_dataset, val_dataset, params, checkpoint_dir)

    model.save(out_path)

    # Evaluate the model
    loss_test = evaluate(model, test_data[:, :5], test_data[:, 5:], batch_size=batch_size)
    print(f"test_loss {loss_test}")

if __name__ == '__main__':
    h5_filename = r"C:\Users\proxi\Documents\coding\TOF_ML\src\simulations\combined_data.h5"
    run_train("/Users/proxi/Documents/coding/TOF_ML/stored_models/test_001/13",
              {"layer_size": 128, "batch_size": 1024, 'dropout': 0.2,
               'learning_rate': 0.001, 'optimizer': 'RMSprop'},
              h5_filename, r"C:\Users\proxi\Documents\coding\TOF_ML\stored_models\test_001\13\checkpoints")
