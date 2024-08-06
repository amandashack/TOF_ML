"""
This file is a sandbox for testing/running functions
"""
from plotter import one_plot_multi_scatter, pass_versus_counts
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
import numpy as np
from model_gen import train_main_model
from sklearn.model_selection import KFold
import sys
import os
import tensorflow as tf
from model_eval import evaluate
import re
from loaders.load_and_save import load_from_h5, DataGenerator, DataGeneratorTofToKE
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
        generator = DataGeneratorTofToKE(data, None, batch_size=len(data))
        data_with_interactions = generator.calculate_interactions(data[:, :4])
        all_data = np.column_stack([data_with_interactions, data[:, 4]])
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


def run_train(out_path, params, h5_filename, checkpoint_dir, n_splits=2):
    # Load data from the HDF5 file
    with h5py.File(h5_filename, 'r') as hf:
        data = hf["data"][:]
        # log of TOF and log of KE
        data[:, 0] = np.log(data[:, 0] + data[:, 2])  # initial kinetic energy (was in pass energy)
        data[:, 5] = np.log(data[:, 5])
        # mask the data for this model
        mask = data[:, -1]
        data = data[mask]

    # select only the necessary data for this model
    print(data.shape, data[:10, :])
    data_lean = np.delete(data, 1, axis=1)
    print(data_lean.shape, data_lean[:10, :])

    # Split data into train and test sets
    train_data, test_data = partition_data(data_lean)

    # Save test data
    save_test_data(test_data, out_path)

    # Define the path for scalers
    scalers_path = os.path.join(out_path, 'scalers.pkl')

    # Calculate or load scalers
    scalers = calculate_scalers(train_data, scalers_path)

    # Define batch size
    batch_size = int(params["batch_size"])

    # Prepare cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    for train_index, val_index in kfold.split(train_data):
        train_fold_data = train_data[train_index]
        val_fold_data = train_data[val_index]

        # Calculate steps per epoch
        params['steps_per_epoch'] = np.ceil(len(train_fold_data) / batch_size).astype(int)
        params['validation_steps'] = np.ceil(len(val_fold_data) / batch_size).astype(int)

        # Create generator instances for training and validation datasets using veto model
        train_gen = DataGeneratorTofToKE(train_fold_data, scalers, batch_size=batch_size)
        val_gen = DataGeneratorTofToKE(val_fold_data, scalers, batch_size=batch_size)

        # Create tf.data.Dataset from the generators
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, output_signature=(
                tf.TensorSpec(shape=(None, 14), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            )
        ).take(len(train_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_gen, output_signature=(
                tf.TensorSpec(shape=(None, 14), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            )
        ).take(len(val_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

        # Train the main model
        print(f"Training main model on fold {fold}/{n_splits}...")
        model, history = train_main_model(train_dataset, val_dataset, params, checkpoint_dir)

        model.save(os.path.join(out_path, f"main_model_fold_{fold}.h5"))
        with open(os.path.join(out_path, f"history_fold_{fold}.pkl"), 'wb') as f:
            pickle.dump(history.history, f)

        print(f"Models and history for fold {fold} saved.")
        fold += 1

    # Evaluate the final model on the test set
    test_gen = DataGenerator(test_data, scalers, batch_size=batch_size)
    test_dataset = tf.data.Dataset.from_generator(
        test_gen, output_signature=(
            tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
        )
    ).take(len(test_data)).cache().prefetch(tf.data.experimental.AUTOTUNE)

    loss_test = model.evaluate(test_dataset, batch_size=batch_size)
    print(f"Test loss: {loss_test}")

if __name__ == '__main__':
    h5_filename = r"C:\Users\proxi\Documents\coding\TOF_ML_backup\src\simulations\combined_data_shuffled.h5"
    run_train("/Users/proxi/Documents/coding/stored_models/test_001/21",
              {"layer_size": 32, "batch_size": 256, 'dropout': 0.2,
               'learning_rate': 0.1, 'optimizer': 'RMSprop'},
              h5_filename, r"C:\Users\proxi\Documents\coding\stored_models\test_001\21\checkpoints")
