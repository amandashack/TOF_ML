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
    """
    Finds the latest checkpoint directory in the checkpoint directory.
    Assumes checkpoint directories are named as 'main_cp-XXXX' where XXXX is the epoch number.
    """
    list_of_dirs = glob.glob(os.path.join(checkpoint_dir, "main_cp-*"))
    if not list_of_dirs:
        return None
    # Extract epoch numbers and find the highest
    latest_dir = None
    latest_epoch = -1
    for dir_path in list_of_dirs:
        basename = os.path.basename(dir_path)
        match = re.match(r"main_cp-(\d+)", basename)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_dir = dir_path
    return latest_dir

def is_chief(strategy):
    task_type, task_id = (strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)
    return task_type == 'chief'

def train_tof_to_energy_model(model, latest_checkpoint, dataset_train, dataset_val, params,
                              checkpoint_dir, param_ID, job_name, meta_file, strategy):
    # Learning rate scheduler and early stopping
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    # Define log directory for TensorBoard (only chief worker)
    #if is_chief(strategy)
    if True:
        log_dir = os.path.join(checkpoint_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='batch',
            profile_batch='500,520'  # Enable profiling for batches 500 to 520
        )
        checkpoint_path = os.path.join(checkpoint_dir, "main_cp-{epoch:04d}")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=False,  # Save the entire model including optimizer state
            save_freq='epoch',  # Ensure checkpoints are saved at the end of each epoch
            save_format='tf',  # Use the TensorFlow SavedModel format
            verbose=1
        )
    else:
        tensorboard_callback = None
        checkpoint = None

    # Callbacks list
    callbacks = [reduce_lr, early_stop]
    if checkpoint:
        callbacks.append(checkpoint)
    if tensorboard_callback:
        callbacks.append(tensorboard_callback)

    if not os.path.exists(checkpoint_dir): # and is_chief(strategy):
        os.makedirs(checkpoint_dir)

    if latest_checkpoint:
        # Extract the epoch number from the checkpoint directory name
        checkpoint_pattern = r"main_cp-(\d+)"
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

    epochs = params.get('epochs', 10)
    steps_per_epoch = params.get('steps_per_epoch', 10)
    validation_steps = params.get('validation_steps', 10)

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        initial_epoch=initial_epoch
    )

    print(f"Early stopping at epoch {early_stop.stopped_epoch}")
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"Best epoch based on validation loss: {best_epoch}")
    return model, history

# Training function
def train_model(data_filepath, model_outpath, params, param_ID, job_name, sample_size=None):
    # Initialize the strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Obtain task information from TF_CONFIG
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    task = tf_config.get('task', {})
    print('task ', task)
    task_type = task.get('type', 'worker')
    print(task_type)
    task_index = task.get('index', 0)
    print(task_index)
    cluster_spec = tf_config.get('cluster', {})
    num_workers = len(cluster_spec.get('worker', [])) + 1 # Include chief in the count
    print(num_workers)
    worker_index = task_index

    # Define the meta file path
    meta_file = os.path.join(os.path.dirname(model_outpath), 'meta.txt')
    scalers_path = os.path.join(model_outpath, 'scalers.pkl')
    indices_path = os.path.join(model_outpath, 'data_indices.npz')
    checkpoint_dir = os.path.join(model_outpath, "checkpoints")

    per_worker_batch_size = params.get('batch_size', 512)
    global_batch_size = per_worker_batch_size * num_workers
    print(global_batch_size)

    # Load or create data indices
    if os.path.exists(indices_path):
        # Load existing indices
        print("Loading existing data indices...")
        indices_data = np.load(indices_path)
        partition = {
            'train': indices_data['train_indices'],
            'validation': indices_data['validation_indices'],
            'test': indices_data['test_indices']
        }
    else:
        # Load the dataset length from the file
        with h5py.File(data_filepath, 'r') as hf:
            data_len = len(hf['combined_data'])

        # Create partition dictionary for train, validation, and test sets (80/10/10 split)
        indices = np.arange(data_len)
        partition = {
            'train': indices[:int(0.8 * data_len)],
            'validation': indices[int(0.8 * data_len):int(0.9 * data_len)],
            'test': indices[int(0.9 * data_len):]
        }

        # Save indices
        np.savez(indices_path,
                 train_indices=partition['train'],
                 validation_indices=partition['validation'],
                 test_indices=partition['test'])
        print(f"Data indices saved to {indices_path}")

    # Apply sampling if a sample size is provided
    if sample_size:
        def sample_indices(indices, size):
            if len(indices) > size:
                return random.sample(list(indices), size)
            return indices  # Return all if the requested sample size exceeds available data

        partition['train'] = sample_indices(partition['train'], sample_size)
        partition['validation'] = sample_indices(partition['validation'], int(0.1 * sample_size))
        partition['test'] = sample_indices(partition['test'], int(0.1 * sample_size))

    # Calculate steps per epoch based on sharded data
    sharded_train_size = len(partition['train']) // num_workers
    sharded_validation_size = len(partition['validation']) // num_workers

    params['steps_per_epoch'] = math.ceil(sharded_train_size / global_batch_size)
    params['validation_steps'] = math.ceil(sharded_validation_size / global_batch_size)
    params['steps_per_execution'] = max(params['steps_per_epoch'] // 10, 1)  # Ensure at least 1

    print(
        f"Worker {worker_index}: steps_per_epoch={params['steps_per_epoch']}, "
        f"steps_per_execution={params['steps_per_execution']}")

    # Calculate min and max values for scaling
    min_values, max_values = calculate_and_save_scalers(partition['train'], data_filepath, scalers_path)

    with strategy.scope():
        dataset_train = create_dataset(partition['train'], data_filepath, global_batch_size, num_workers, worker_index,
                                       shuffle=True)
        dataset_val = create_dataset(partition['validation'], data_filepath, global_batch_size, num_workers,
                                     worker_index, shuffle=True)

        # Check for existing checkpoints
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Loading model from checkpoint: {latest_checkpoint}")
            model = tf.keras.models.load_model(latest_checkpoint, custom_objects={
                'LogTransformLayer': LogTransformLayer,
                'InteractionLayer': InteractionLayer,
                'ScalingLayer': ScalingLayer,
                'TofToEnergyModel': TofToEnergyModel
            })
        else:
            print("No checkpoint found. Initializing a new model.")
            model = TofToEnergyModel(params, min_values, max_values)

    start_time = time.time()
    # Train the main model
    model, history = train_tof_to_energy_model(model, latest_checkpoint, dataset_train, dataset_val, params,
                                               checkpoint_dir, param_ID, job_name, meta_file, strategy)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Save the model (ensure only the chief worker saves the model)
    #if is_chief(strategy):
    if True:
        model.save(os.path.join(model_outpath, "main_model"), save_format="tf")
        print("Model saved by the chief worker.")

    # Optionally, handle evaluation on the test set
    # dataset_test = create_dataset(partition['test'], data_filepath, global_batch_size, shuffle=False)
    # loss_test = model.evaluate(dataset_test, verbose=0)
    # print(f"Test loss: {loss_test}")

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
if __name__ == '__main__':
    model_outpath = r"C:\Users\proxi\Documents\coding\stored_models\test_001\36"
    data_filepath = r"C:\Users\proxi\Documents\coding\TOF_data\TOF_data\combined_data.h5"
    params = {
        "layer_size": 32,
        "batch_size": int(1024/2),
        "dropout": 0.2,
        "learning_rate": 0.1,
        "optimizer": 'RMSprop',
        "job_name": "default_deep",
        "epochs": 10  # Add epochs parameter if needed
    }
    train_model(data_filepath, model_outpath, params, 12, 'default', sample_size=200000)


"""if __name__ == '__main__':
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
    train_model(DATA_FILENAME, output_file_path, params, param_ID, job_name)"""
