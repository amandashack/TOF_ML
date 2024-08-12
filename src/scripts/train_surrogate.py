import os
import pickle
import numpy as np
import tensorflow as tf
import glob
from sklearn.model_selection import KFold
import h5py
import sys
from training_functions import *
sys.path.insert(0, os.path.abspath('..'))
from loaders.load_and_save import DataGenerator, DataGeneratorWithVeto
from models.surrogate_model import train_main_model
from models.veto_model import train_veto_model

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


def run_train(out_path, params, h5_filename, checkpoint_dir, n_splits=2, subset_percentage=None):
    # Load data from the HDF5 file
    with h5py.File(h5_filename, 'r') as hf:
        # preprocess data
        data = hf["data"][:]
        # log of TOF and log of KE
        data[:, 0] = np.log(data[:, 0] + data[:, 2])  # initial kinetic energy (was in pass energy)
        data[:, 5] = np.log(data[:, 5])

    # Split data into train and test sets
    train_data, test_data = partition_data(data)

    # Optionally select a subset of the training data
    if subset_percentage is not None and 0 < subset_percentage < 1:
        original_size = len(train_data)
        subset_size = int(original_size * subset_percentage)
        train_data = train_data[np.random.choice(original_size, subset_size, replace=False)]
        print(f"Original training data size: {original_size}")
        print(f"Subset training data size: {subset_size}")

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

        # Check if veto model exists
        veto_model = load_veto_model_if_exists(checkpoint_dir, fold)

        if veto_model is None:
            # If no veto model exists, create generator instances and train a new veto model
            veto_train_gen = DataGenerator(train_fold_data, scalers, batch_size=batch_size)
            veto_val_gen = DataGenerator(val_fold_data, scalers, batch_size=batch_size)

            veto_train_dataset = tf.data.Dataset.from_generator(
                veto_train_gen, output_signature=(
                    tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # mask as the target
                )
            ).take(len(train_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

            veto_val_dataset = tf.data.Dataset.from_generator(
                veto_val_gen, output_signature=(
                    tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # mask as the target
                )
            ).take(len(val_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

            print(f"Training veto model on fold {fold}/{n_splits}...")
            veto_model, history = train_veto_model(veto_train_dataset, veto_val_dataset, params, checkpoint_dir)
            veto_model_path = os.path.join(checkpoint_dir, f"veto_model_fold_{fold}.h5")
            veto_model.save(veto_model_path)

        # Create generator instances for training and validation datasets using veto model
        train_gen = DataGeneratorWithVeto(train_fold_data, scalers, veto_model, batch_size=batch_size)
        val_gen = DataGeneratorWithVeto(val_fold_data, scalers, veto_model, batch_size=batch_size)

        # Create tf.data.Dataset from the generator
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, output_signature=(
                tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
            )
        ).take(len(train_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_gen, output_signature=(
                tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
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


if __name__ == '__main__':
    h5_filename = r"C:\Users\proxi\Documents\coding\TOF_ML_backup\src\simulations\combined_data_shuffled.h5"
    run_train("/Users/proxi/Documents/coding/stored_models/test_001/25",
              {"layer_size": 32, "batch_size": 1024, 'dropout': 0.2,
               'learning_rate': 0.4, 'optimizer': 'RMSprop'},
              h5_filename, r"C:\Users\proxi\Documents\coding\stored_models\test_001\25\checkpoints",
              subset_percentage=0.5)