from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import sys
import re
import h5py
import os
from loaders import DataGenerator, DataGeneratorWithVeto, DataGeneratorTofToKE
from models import (train_veto_model, train_main_model, create_main_model,
                    train_tof_to_energy_model, create_tof_to_energy_model)
from scripts import random_sample_data


# TODO: put these in a json or something
DATA_FILENAME = r"C:\Users\proxi\Documents\coding\TOF_data\TOF_data\combined_data.h5"
VETO_MODEL = r"C:\Users\proxi\Documents\coding\TOF_ML\stored_models\surrogate\veto_model.h5"


# Check GPU availability and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus = None
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        pass


def train_model(data_filepath, model_outpath, params, sample_size=100000):
    checkpoint_dir = os.path.join(model_outpath, "checkpoints")
    combined_model_path = os.path.join(model_outpath, "combined_model.h5")

    # Use batch_size from params, default to 256 if not specified
    batch_size = params.get('batch_size', 256)
    scalers_path = os.path.join(model_outpath, 'scalers.pkl')

    # Load the dataset length from the file
    with h5py.File(data_filepath, 'r') as hf:
        data_len = len(hf['combined_data'])

    # Create partition dictionary for train and validation sets (80/20 split)
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

    # Save test data
    test_data_filename = os.path.join(model_outpath, 'test_data.h5')
    test_data_indices = partition['test']
    # Sort the indices
    test_data_indices_sorted = np.sort(test_data_indices)

    with h5py.File(data_filepath, 'r') as hf_in:
        test_data = hf_in['combined_data'][test_data_indices_sorted]
        with h5py.File(test_data_filename, 'w') as hf_out:
            hf_out.create_dataset('test_data', data=test_data)

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

    # Combine models by averaging their weights (cross-validation logic)
    steps_per_execution = params['steps_per_epoch']//10
    #combined_model = train_tof_to_energy_model(params, steps_per_execution=steps_per_execution)  # Create a new model instance
    #fold_models = [model]  # Assuming cross-validation, add models to this list
    #combined_weights = [m.get_weights() for m in fold_models]
    #new_weights = []

    #for weights_tuple in zip(*combined_weights):
    #    new_weights.append([np.mean(np.array(w), axis=0) for w in zip(*weights_tuple)])

    #combined_model.set_weights(new_weights)
    #combined_model.save(combined_model_path)

    # Final evaluation on test data
    with h5py.File(test_data_filename, 'r') as hf:
        test_data = hf['test_data'][:]

    test_gen = DataGenerator(
        list_IDs=range(len(test_data)),
        labels=np.ones(len(test_data)),  # Dummy labels
        data_filename=data_filepath,
        batch_size=batch_size,
        dim=(14,),
        shuffle=False,
        scalers_path=scalers_path
    )

    loss_test = model.evaluate(test_gen, steps=len(test_gen))
    print(f"Final test loss: {loss_test}")


"""if __name__ == '__main__':
    # Collect parameters passed through command line
    output_file_path = sys.argv[1]
    params_str = ' '.join(sys.argv[2:])
    params_list = re.findall(r'(\w+)=(\S+)', params_str)
    params = {}
    for key, value in params_list:
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value

    # Call the training function with parsed parameters
    train_model(DATA_FILENAME, output_file_path, params)"""
if __name__ == '__main__':
    h5_filename = r"C:\Users\proxi\Documents\coding\TOF_data\TOF_data\combined_data.h5"
    train_model(DATA_FILENAME, "/Users/proxi/Documents/coding/stored_models/test_001/28",
              {"layer_size": 64, "batch_size": 256, 'dropout': 0.2,
               'learning_rate': 0.1, 'optimizer': 'RMSprop'}, sample_size=None)
