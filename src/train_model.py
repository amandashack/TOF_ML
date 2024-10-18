# src/train_model.py

import numpy as np
import math
import h5py
import os
import sys
import json
import re
import glob
import random
import datetime
import pickle
import argparse
import tensorflow as tf
import time
from loaders import calculate_and_save_scalers, create_dataset
from models import TofToEnergyModel, LogTransformLayer, ScalingLayer, InteractionLayer


# Check GPU availability and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        pass


def get_latest_checkpoint(checkpoint_dir):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    return latest_checkpoint


def train_tof_to_energy_model(model, latest_checkpoint, dataset_train, dataset_val, params,
                              checkpoint_dir, param_ID, job_name, meta_file, strategy):
    # Learning rate scheduler and early stopping
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    # Define log directory for TensorBoard
    log_dir = os.path.join(checkpoint_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        profile_batch='100,120'  # Enable profiling for batches 100 to 120
    )
    checkpoint_path = os.path.join(checkpoint_dir, "main_cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )

    # Callbacks list
    callbacks = [reduce_lr, early_stop, checkpoint, tensorboard_callback]

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if latest_checkpoint:
        # Extract the epoch number from the checkpoint file name
        checkpoint_pattern = r"main_cp-(\d{4})\.ckpt"
        match = re.search(checkpoint_pattern, os.path.basename(latest_checkpoint))
        if match:
            initial_epoch = int(match.group(1))
            print(f"Resuming training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("Could not extract epoch number from checkpoint. Starting from epoch 0.")
    else:
        initial_epoch = 0
        print("No checkpoint found. Starting from epoch 0.")

    epochs = params.get('epochs', 200)
    steps_per_epoch = params.get('steps_per_epoch', 10)
    validation_steps = params.get('validation_steps', 10)

    start_time = time.time()

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        initial_epoch=initial_epoch
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    print(f"Early stopping at epoch {early_stop.stopped_epoch}")
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"Best epoch based on validation loss: {best_epoch}")
    return model, history

# Training function
def train_model(data_filepath, model_outpath, params, param_ID, job_name, sample_size=None):
    # Initialize the strategy
    strategy = tf.distribute.MirroredStrategy()

    # Define the meta file path
    meta_file = os.path.join(os.path.dirname(model_outpath), 'meta.txt')
    scalers_path = '/sdf/scratch/users/a/ajshack/scalers.pkl'  # Path to where scalers are saved
    indices_path = '/sdf/scratch/users/a/ajshack/data_indices.npz'  # Path to where indices are saved
    checkpoint_dir = os.path.join(model_outpath, "checkpoints")

    batch_size = params['batch_size']

    # Load existing indices
    print("Loading existing data indices...")
    indices_data = np.load(indices_path)
    partition = {
        'train': indices_data['train_indices'],
        'validation': indices_data['validation_indices'],
        'test': indices_data['test_indices']
    }

    # Apply sampling if a sample size is provided
    if sample_size:
        def sample_indices(indices, size):
            if len(indices) > size:
                return random.sample(list(indices), size)
            return indices  # Return all if the requested sample size exceeds available data

        partition['train'] = sample_indices(partition['train'], sample_size)
        partition['validation'] = sample_indices(partition['validation'], int(0.1 * sample_size))
        partition['test'] = sample_indices(partition['test'], int(0.1 * sample_size))

    params['steps_per_epoch'] = math.ceil(len(partition['train']) / params['batch_size'])
    params['validation_steps'] = math.ceil(len(partition['validation'])  / params['batch_size'])
    params['steps_per_execution'] = max(params['steps_per_epoch'] // 10, 1)

    # Load scalers
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
        min_values = scalers['min_values']
        max_values = scalers['max_values']
        print(f"Scalers loaded from {scalers_path}")

    with strategy.scope():
        dataset_train = create_dataset(partition['train'], data_filepath, batch_size,
                                       shuffle=True)
        dataset_val = create_dataset(partition['validation'], data_filepath, batch_size,
                                     shuffle=True)

        # Check for existing checkpoints
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Loading model weights from checkpoint: {latest_checkpoint}")
            model = TofToEnergyModel(params, min_values, max_values)
            model.load_weights(latest_checkpoint)
        else:
            print("No checkpoint found. Initializing a new model.")
            model = TofToEnergyModel(params, min_values, max_values)

    # Train the main model
    model, history = train_tof_to_energy_model(model, latest_checkpoint, dataset_train, dataset_val, params,
                                               checkpoint_dir, param_ID, job_name, meta_file, strategy)

    # Save the entire model after training
    model.save(os.path.join(model_outpath, "main_model"), save_format="tf")
    print("Model saved.")

    # Optionally, handle evaluation on the test set
    # dataset_test = create_dataset(partition['test'], data_filepath, global_batch_size, shuffle=False)
    # loss_test = model.evaluate(dataset_test, verbose=0)
    # print(f"Test loss: {loss_test}")

DATA_FILENAME = "/sdf/scratch/users/a/ajshack/combined_data_large.h5"


# Entry point with argument parsing
"""if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the TofToEnergyModel with distributed strategy.')
    parser.add_argument('--model_outpath', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--data_filepath', type=str, required=True, help='Path to the HDF5 data file.')
    parser.add_argument('--param_ID', type=int, required=True, help='Parameter ID for the run.')
    parser.add_argument('--job_name', type=str, required=True, help='Job name for model configuration.')
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size for training.')

    args = parser.parse_args()

    # Load params from environment variable if set, else use defaults
    params_json = os.getenv('PARAMS')
    if params_json:
        params = json.loads(params_json)
    else:
        params = {
            "layer_size": 32,
            "batch_size": 512,  # Changed to 512 (1024 / 2)
            "dropout": 0.2,
            "learning_rate": 0.1,
            "optimizer": 'RMSprop',
            "job_name": args.job_name,
            "epochs": 10  # Add epochs parameter if needed
        }

    train_model(
        data_filepath=args.data_filepath,
        model_outpath=args.model_outpath,
        params=params,
        param_ID=args.param_ID,
        job_name=args.job_name,
        sample_size=args.sample_size
    )"""


# Entry point
#if __name__ == '__main__':
#    model_outpath = r"C:\Users\proxi\Documents\coding\stored_models\test_001\37"
#    data_filepath = r"C:\Users\proxi\Documents\coding\TOF_data\TOF_data\combined_data.h5"
#    params = {
#        "layer_size": 32,
#        "batch_size": int(1024/2),
#        "dropout": 0.2,
#        "learning_rate": 0.1,
#        "optimizer": 'RMSprop',
#        "job_name": "default_deep",
#        "epochs": 5  # Add epochs parameter if needed
#    }
#    train_model(data_filepath, model_outpath, params, 12, 'default', sample_size=200000)


if __name__ == '__main__':
    # Collect parameters passed through command line
    output_file_path = sys.argv[1]
    job_name = sys.argv[2]
    params_str = ' '.join(sys.argv[3:])
    params_list = re.findall(r'--?(\w+)=(\S+)', params_str)
    params = {}
    for key, value in params_list:
        try:
            # Convert to float or int if possible
            if '.' in value or 'e' in value.lower():
                params[key] = float(value)
            else:
                params[key] = int(value)
        except ValueError:
            params[key] = value

    # Extract param_ID from output_file_path
    param_ID = os.path.basename(output_file_path)
    params['job_name'] = job_name

    # Call the training function with parsed parameters
    train_model(DATA_FILENAME, output_file_path, params, param_ID, job_name)
