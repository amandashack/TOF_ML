from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import sys
import re
import h5py
import os
import random
from loaders import DataGenerator, DataGeneratorWithVeto, DataGeneratorTofToKE
from models import (train_veto_model, train_main_model, create_main_model,
                    train_tof_to_energy_model, create_tof_to_energy_model)
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
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        pass


def train_model(data_filepath, model_outpath, params, sample_size=None):
    checkpoint_dir = os.path.join(model_outpath, "checkpoints")

    # Use batch_size from params, default to 256 if not specified
    batch_size = params.get('batch_size', 256)
    scalers_path = os.path.join(model_outpath, 'scalers.pkl')

    indices_path = os.path.join(model_outpath, 'data_indices.npz')
    test_data_filename = os.path.join(model_outpath, 'test_data.h5')

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
        if sample_size:
            indices = random_sample_data(indices, sample_size=sample_size)
        data_len = len(indices)
        split_index1 = int(0.8 * data_len)
        split_index2 = int(0.8 * split_index1)
        print(split_index2, split_index1, data_len)

        partition = {
            'train': indices[:split_index2],
            'validation': indices[split_index2:split_index1],
            'test': indices[split_index1:]
        }

        # Save indices
        np.savez(indices_path,
                 train_indices=partition['train'],
                 validation_indices=partition['validation'],
                 test_indices=partition['test'])
        print(f"Data indices saved to {indices_path}")

    # Save test data if it doesn't exist
    if not os.path.exists(test_data_filename):
        print("Saving test data...")
        test_data_indices = partition['test']
        # Sort the indices
        test_data_indices_sorted = np.sort(test_data_indices)

        with h5py.File(data_filepath, 'r') as hf_in:
            test_data = hf_in['combined_data'][test_data_indices_sorted]
            with h5py.File(test_data_filename, 'w') as hf_out:
                hf_out.create_dataset('test_data', data=test_data)
        print(f"Test data saved to {test_data_filename}")
    else:
        print("Test data already exists. Skipping saving test data.")
        with h5py.File(test_data_filename, 'r') as hf:
            test_data = hf['test_data'][:]

    # Initialize the training and validation generators
    train_gen = DataGeneratorTofToKE(
        list_IDs=partition['train'],
        labels=np.ones(len(partition['train'])),  # Dummy labels, adjust as needed
        data_filename=data_filepath,
        batch_size=batch_size,
        dim=(14,),  # Adjust dimension as needed
        shuffle=True,
        scalers_path=scalers_path  # Pass scalers_path to use or calculate scalers
    )

    train_gen.calculate_scalers()

    val_gen = DataGeneratorTofToKE(
        list_IDs=partition['validation'],
        labels=np.ones(len(partition['validation'])),  # Dummy labels, adjust as needed
        data_filename=data_filepath,
        batch_size=batch_size,
        dim=(14,),  # Adjust dimension as needed
        shuffle=True,
        scalers_path=scalers_path  # Pass scalers_path to use or calculate scalers
    )

    # Calculate steps per epoch
    params['steps_per_epoch'] = len(train_gen)
    params['validation_steps'] = len(val_gen)

    val_gen.calculate_scalers()

    """# Check if veto model exists
    try:
        veto_model = tf.keras.models.load_model(VETO_MODEL)
    except:
        veto_model = None

    if veto_model is None:
        # Training veto model
        veto_model, history = train_veto_model(train_gen, val_gen, params, checkpoint_dir)
        veto_model_path = os.path.join(checkpoint_dir, "veto_model.h5")
        veto_model.save(veto_model_path)"""

    # Train the main model
    model, history = train_tof_to_energy_model(train_gen, val_gen, params, checkpoint_dir)

    model.save(os.path.join(model_outpath, "main_model.h5"))

    test_gen = DataGeneratorTofToKE(
        list_IDs=range(len(test_data)),
        labels=np.ones(len(test_data)),  # Dummy labels
        data_filename=data_filepath,
        batch_size=batch_size,
        dim=(14,),
        shuffle=False,
        scalers_path=scalers_path
    )

    loss_test = model.evaluate(test_gen, steps=len(test_gen), verbose=0)
    print(f"test_loss {loss_test}")


if __name__ == '__main__':
    # Collect parameters passed through command line
    output_file_path = sys.argv[1]
    params_str = ' '.join(sys.argv[2:])
    params_list = re.findall(r'(\w+)=(\S+)', params_str)
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

    # Call the training function with parsed parameters
    train_model(DATA_FILENAME, output_file_path, params)

if __name__ == '__main__':
    data_filepath = r"C:\Users\proxi\Documents\coding\TOF_data\TOF_data\combined_data.h5"
    model_outpath = r"C:\Users\proxi\Documents\coding\stored_models\test_001\28"
    params = {
        "layer_size": 64,
        "batch_size": 256,
        "dropout": 0.2,
        "learning_rate": 0.1,
        "optimizer": 'RMSprop',
        "epochs": 200  # Add epochs parameter if needed
    }
    train_model(data_filepath, model_outpath, params, sample_size=None)
