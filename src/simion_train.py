"""
This file is a sandbox for testing/running functions
"""
from plotter import one_plot_multi_scatter, pass_versus_counts
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from model_gen import run_model
import sys
import os
import tensorflow as tf
from model_eval import evaluate
import re
from loaders.load_and_save import load_from_h5


def y0_fit(x_value):
    y0 = -0.4986 * x_value - 0.5605
    return y0


def preprocess_features(time_of_flight, retardation, mid1, mid2, kinetic_energy):
    # these are already log
    log_tof = time_of_flight
    log_ke = np.log(kinetic_energy)
    residual = log_tof - y0_fit(log_ke)

    return np.column_stack((log_tof, residual, retardation, mid1, mid2)), log_ke

def run_train(out_path, params, h5_filename):
    """
    Load data from an HDF5 file, split it into training, validation, and test sets,
    and train the model.

    Parameters:
    out_path (str): The path to save the trained model.
    params (dict): Parameters for the model.
    h5_filename (str): The name of the HDF5 file containing the data.
    """
    # Load the combined data from the HDF5 file
    combined_array = load_from_h5(h5_filename)
    nan_mask = np.isnan(combined_array).any(axis=1)
    cleaned_combined = combined_array[~nan_mask]

    # Split the data into inputs and outputs
    inputs = cleaned_combined[:, :-1]  # [TOF, retardation, mid1, mid2]
    outputs = cleaned_combined[:, -1]  # [kinetic_energy]

    inputs_preprocessed, outputs_preprocessed = preprocess_features(inputs[:, 0], inputs[:, 1], inputs[:, 2],
                                                                    inputs[:, 3], outputs)

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("Mismatch between number of samples in inputs and outputs")

    # Shuffle the data
    inputs_preprocessed, outputs_preprocessed = shuffle(inputs_preprocessed, outputs_preprocessed,
                                                        random_state=42)
    print(np.unique(inputs_preprocessed[:, 2]))

    # Further split the data into training, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(inputs_preprocessed, outputs_preprocessed,
                                                        test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Verify data shapes
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Train the model
    model, history = run_model(np.array(x_train), np.array(y_train),
                               np.array(x_val), np.array(y_val), params)

    model.save(out_path)

    loss_test = evaluate(model, x_test, y_test, plot=True)

    print(f"test_loss {loss_test}")

if __name__ == '__main__':
    #p = ' '.join(sys.argv[2:])
    #p = re.findall(r'(\w+)=(\S+)', p)
    #params = dict((p[i][0], p[i][1]) for i in range(len(p)))
    #output_file_path = sys.argv[1]
    #run_train(output_file_path, params)
    h5_filename = r"C:\Users\proxi\Documents\coding\TOF_ML\src\simulations\tof_to_energy_data.h5"
    run_train("/Users/proxi/Documents/coding/TOF_ML/stored_models/test_001/13",
              {"layer_size": 64, "batch_size": 256, 'dropout': 0.2,
               'learning_rate': 0.0001, 'optimizer': 'RMSprop'}, h5_filename)
