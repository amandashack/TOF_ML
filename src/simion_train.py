"""
This file is a sandbox for testing/running functions
"""
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from model_gen import run_model
import sys
import os
import tensorflow as tf
from model_eval import evaluate
import re
from loaders.load_and_save import load_from_h5
import joblib


def preprocess_features(data):
    # these are already log
    ke = 0
    ele = 1
    ret = 2
    v1 = 3
    v2 = 4
    tof = 5
    ypos = 6
    m = 7
    print(data.shape)
    ATOL = 1e-2      # Absolute tolerance for floating-point comparison

    # Create boolean masks using np.isclose for v1 and v2
    v1_mask = np.isclose(data[:, v1], 0.11248, atol=ATOL)
    v2_mask = np.isclose(data[:, v2], 0.1354, atol=ATOL)
    print(v1_mask[:10], v2_mask[:10], data[:10].tolist())
    #ret_mask = np.where(data[:, ret] >= 0, 1, 0)
    vmask = data[:, m].astype(bool)
    combined_mask = v1_mask & v2_mask & vmask #& ret_mask
    filtered_data = data[combined_mask]
    print(filtered_data[:10].tolist())
    X = filtered_data[:, [ret, tof]]
    # Apply interaction terms
    x1 = X[:, 0]
    x2 = np.log2(X[:, 1])
    interaction_terms = np.column_stack([
        x1 * x2,
        x1 ** 2,
        x2 ** 2,
        ])
    processed_input = np.hstack([x1.flatten(), x2.flatten(), interaction_terms])
    # Extract output: log(Kinetic Energy)
    y = np.log(filtered_data[:, ke]).reshape(-1, 1)  # Reshape for scaler
    
    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit scalers on the respective data
    X_scaled = scaler_X.fit_transform(processed_input)
    y_scaled = scaler_y.fit_transform(y).flatten()  # Flatten to 1D array
    
    return X_scaled, y_scaled, scaler_X, scaler_y


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
    inputs_preprocessed, outputs_preprocessed, scaler_x, scaler_y = preprocess_features(cleaned_combined)

    # Shuffle the data
    #inputs_preprocessed, outputs_preprocessed = shuffle(inputs_preprocessed, outputs_preprocessed,
    #                                                    random_state=42)
    

    # Further split the data into training, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(inputs_preprocessed, outputs_preprocessed,
                                                        test_size=0.25, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Verify data shapes
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Train the model
    model, history = run_model(x_train, y_train, x_val, y_val, params)

    # Ensure the output directory exists
    os.makedirs(out_path, exist_ok=True)
    
    # Save the model (assuming TensorFlow/Keras)
    model_save_path = os.path.join(out_path, "saved_model")
    model.save(model_save_path)
    
    # Save the scalers for later inverse transformation
    scaler_path_X = os.path.join(out_path, "scaler_X.joblib")
    scaler_path_y = os.path.join(out_path, "scaler_y.joblib")
    joblib.dump(scaler_x, scaler_path_X)
    joblib.dump(scaler_y, scaler_path_y)

    loss_test = evaluate(model, x_test, y_test, scaler_y, plot=True)

    print(f"test_loss {loss_test}")

if __name__ == '__main__':
    #p = ' '.join(sys.argv[2:])
    #p = re.findall(r'(\w+)=(\S+)', p)
    #params = dict((p[i][0], p[i][1]) for i in range(len(p)))
    #output_file_path = sys.argv[1]
    #run_train(output_file_path, params)
    h5_filename = r"/sdf/scratch/users/a/ajshack/combined_data_large.h5"
    run_train("/sdf/scratch/users/a/ajshack/tmox1016823/model_trials/2",
              {"layer_size": 16, "batch_size": 256, 'dropout': 0.4,
                  'learning_rate': 0.01, 'optimizer': 'RMSprop'}, h5_filename)
