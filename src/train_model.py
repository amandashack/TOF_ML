import tensorflow as tf
import numpy as np
import math
import h5py
import os
import sys
import re
import random
import pickle
from loaders import DataGenerator, DataGeneratorWithVeto, DataGeneratorTofToKE, calculate_and_save_scalers, create_dataset
from models import train_tof_to_energy_model#, create_tof_to_energy_model
from scripts import random_sample_data


# TODO: put these in a json or something
DATA_FILENAME = r"/sdf/home/a/ajshack/combined_data.h5"

seed_value = 42  # Or any other integer
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# Check GPU availability and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Optionally, set visible devices if you want to limit GPU usage
        # tf.config.experimental.set_visible_devices(gpus[:2], 'GPU')  # Use first two GPUs
    except RuntimeError as e:
        print(e)


# Training function
def train_model(data_filepath, model_outpath, params, param_ID, job_name, sample_size=None):
    checkpoint_dir = os.path.join(model_outpath, "checkpoints")

    # Define the meta file path
    meta_file = os.path.join(os.path.dirname(model_outpath), 'meta.txt')
    scalers_path = os.path.join(model_outpath, 'scalers.pkl')
    indices_path = os.path.join(model_outpath, 'data_indices.npz')

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

        # Create partition dictionary for train, validation, and test sets (80/20 split)
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
        partition['validation'] = sample_indices(partition['validation'], int(0.2 * sample_size))
        partition['test'] = sample_indices(partition['test'], int(0.1 * sample_size))

    batch_size = params.get('batch_size', 1024)

    # Calculate steps per epoch
    params['steps_per_epoch'] = math.ceil(len(partition['train']) / batch_size)
    params['validation_steps'] = math.ceil(len(partition['validation']) / batch_size)
    params['steps_per_execution'] = params['steps_per_epoch'] // 10
    print(params['steps_per_epoch'], params['steps_per_execution'])

    # Calculate min and max values for scaling
    min_values, max_values = calculate_and_save_scalers(partition['train'], data_filepath, scalers_path)

    # Create datasets
    dataset_train = create_dataset(partition['train'], data_filepath, batch_size, shuffle=True)
    dataset_val = create_dataset(partition['validation'], data_filepath, batch_size, shuffle=False)

    # Train the main model
    model, history = train_tof_to_energy_model(
        dataset_train, dataset_val, params, checkpoint_dir, param_ID, job_name, meta_file, min_values, max_values
    )

    model.save(os.path.join(model_outpath, "main_model"), save_format="tf")

    # Create test dataset
    dataset_test = create_dataset(partition['test'], data_filepath, batch_size, shuffle=False)

    loss_test = model.evaluate(dataset_test, verbose=0)
    print(f"test_loss {loss_test}")


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

if __name__ == '__main__':
    model_outpath = r"C:\Users\proxi\Documents\coding\stored_models\test_001\30"
    data_filepath = r"C:\Users\proxi\Documents\coding\TOF_data\TOF_data\combined_data.h5"
    params = {
        "layer_size": 64,
        "batch_size": 1024,
        "dropout": 0.2,
        "learning_rate": 0.2,
        "optimizer": 'RMSprop',
        "job_name": "default"
    }
    train_model(data_filepath, model_outpath, params, 12, 'default', sample_size=200000)
