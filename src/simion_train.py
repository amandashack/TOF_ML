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
from loaders.load_and_save import load_from_h5


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

    # Split the data into inputs and outputs
    inputs = combined_array[:, :5]  # [pass_energy, elevation, retardation, mid1_ratio, mid2_ratio]
    outputs = combined_array[:, 5:]  # [time of flight, y_tof, mask]

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("Mismatch between number of samples in inputs and outputs")

    # Further split the data into training, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(inputs, outputs, test_size=0.3, random_state=42)
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

    loss_test = evaluate(model, x_test.T, y_test.T)

    print(f"test_loss {loss_test}")

if __name__ == '__main__':
    #p = ' '.join(sys.argv[2:])
    #p = re.findall(r'(\w+)=(\S+)', p)
    #params = dict((p[i][0], p[i][1]) for i in range(len(p)))
    #output_file_path = sys.argv[1]
    #run_train(output_file_path, params)
    h5_filename = r"C:\Users\proxi\Documents\coding\TOF_ML\src\simulations\combined_data.h5"
    run_train("/Users/proxi/Documents/coding/TOF_ML/stored_models/test_001/13",
              {"layer_size": 64, "batch_size": 256, 'dropout': 0.2,
               'learning_rate': 0.1, 'optimizer': 'RMSprop'}, h5_filename)
