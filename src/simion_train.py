from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from model_gen import run_model
import os
from model_eval import evaluate
from loaders.load_and_save import load_from_h5
import joblib
import csv
import glob


def parse_retardation_from_filename(filename):
    """
    Given a filename like 'NM_neg_0.csv' or 'NM_pos_14.csv', extract the retardation value.
    Assumes the filename structure is:
    'NM_(neg|pos)_{value}.csv'
    """
    base = os.path.basename(filename)  # e.g. NM_neg_0.csv
    name, _ = os.path.splitext(base)  # e.g. NM_neg_0
    parts = name.split('_')
    # parts might look like ["NM", "neg", "0"] or ["NM", "pos", "14"]
    sign_part = parts[1]
    value_part = parts[2]

    # Determine sign from 'neg' or 'pos'
    sign = -1 if sign_part == 'neg' else 1
    # Convert value to int and apply sign
    retardation = sign * int(value_part)
    return retardation


def extract_data_from_csv(csv_filename):
    """
    Extract initial KE, final TOF, final X, and retardation from a single CSV file.
    The CSV is assumed to have pairs of lines per ion:
      - First line of pair: initial parameters (with TOF=0, includes initial KE)
      - Second line of pair: final parameters (with final TOF, final X)
    """
    retardation = parse_retardation_from_filename(csv_filename)

    data_rows = []
    with open(csv_filename, 'r', newline='') as f:
        reader = csv.reader(f)

        # Skip the header line if present (assuming the first line might be header)
        # If the first line is known to be header, uncomment:
        # header = next(reader)

        # We know that each ion is represented by two consecutive lines:
        # first = initial conditions, second = final conditions
        # We'll read in pairs:

        # the lines come in pairs:
        # e.g. first line: ["Ion N", "TOF", "X", "Y", "Z", "Azm", "Elv", "KE"]
        # next line: actual data for Ion 1 initial
        # next pair: Ion 1 final
        # We'll assume we just iterate and pick pairs. The first line of each pair should have KE at the end.

        # According to the example provided, the pattern is:
        #   odd lines: initial conditions (TOF=0)
        #   even lines: final conditions
        #
        # We'll store pairs as we go.

        lines = list(reader)
        # The data lines (after header) come in pairs:
        # initial_line, final_line
        # initial_line example: ['1', '0', '12', '0', '0', '0', '0', '0.001']
        # final_line example:   ['1', '20.22574434', '406.7', '0', '0', '0', '0', '300.0010007']
        #
        # Desired output: [initial_KE, final_TOF, final_X, retardation]

        # initial_KE is last element of initial line
        # final_TOF is second element of final line
        # final_X is third element of final line

        # The lines appear to be in pairs:
        # line0: Initial conditions for Ion 1
        # line1: Final conditions for Ion 1
        # line2: Initial conditions for Ion 2
        # line3: Final conditions for Ion 2
        # etc.

        # We'll iterate in steps of 2
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break  # In case of odd number of lines
            initial_line = lines[i]
            final_line = lines[i + 1]

            # Extract values
            initial_KE = float(initial_line[-1])
            final_TOF = float(final_line[1])
            final_X = float(final_line[2])

            # Append row
            data_rows.append([final_TOF, retardation, initial_KE, final_X])

    return data_rows


def build_feature_array(folder_path):
    """
    Go through all CSV files in the given folder (e.g., *.csv),
    read their data, and combine into a single NumPy array.
    Each row: [initial_KE, final_TOF, final_X, retardation]
    """
    all_data = []
    for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
        file_data = extract_data_from_csv(csv_file)
        all_data.extend(file_data)
    all_data = np.array(all_data)
    d_mask = all_data[:, -1] > 406
    data_masked = all_data[d_mask]
    return data_masked

def mask_features(data):
    # these are already log
    ke = 0
    ele = 1
    ret = 2
    v1 = 3
    v2 = 4
    tof = 5
    ypos = 6
    m = 7
    ATOL = 1e-2      # Absolute tolerance for floating-point comparison

    # Create boolean masks using np.isclose for v1 and v2
    v1_mask = np.isclose(data[:, v1], 0.11248, atol=ATOL)
    v2_mask = np.isclose(data[:, v2], 0.1354, atol=ATOL)
    #ret_mask = np.where(data[:, ret] >= 0, 1, 0)
    vmask = data[:, m].astype(bool)
    combined_mask = v1_mask & v2_mask & vmask #& ret_mask
    filtered_data = data[combined_mask]
    X = filtered_data[:, [tof, ret]]
    needed_data = np.column_stack([X, filtered_data[:, ke]])
    return needed_data

def preprocess_data(data):
    X = data[:, [0, 1]]
    # Apply interaction terms
    x1 = np.log2(X[:, 0])
    x2 = X[:, 1]
    interaction_terms = np.column_stack([
        x1 * x2,
        x1 ** 2,
        x2 ** 2,
        ])
    processed_input = np.hstack([x1.flatten(), x2.flatten(), interaction_terms])
    # Extract output: log(Kinetic Energy)
    y = np.log(data[:, 2]).reshape(-1, 1)  # Reshape for scaler
    
    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit scalers on the respective data
    X_scaled = scaler_X.fit_transform(processed_input)
    y_scaled = scaler_y.fit_transform(y).flatten()  # Flatten to 1D array
    
    return X_scaled, y_scaled, scaler_X, scaler_y


def run_train(out_path, params, filename, is_h5=False):
    """
    Load data from an HDF5 file, split it into training, validation, and test sets,
    and train the model.

    Parameters:
    out_path (str): The path to save the trained model.
    params (dict): Parameters for the model.
    h5_filename (str): The name of the HDF5 file containing the data.
    """
    # Load the combined data from the HDF5 file
    if is_h5:
        combined_array = load_from_h5(filename)
        masked_array = mask_features(combined_array)
    else:
        masked_array = build_feature_array(filename)
    nan_mask = np.isnan(masked_array).any(axis=1)
    cleaned_combined = masked_array[~nan_mask]

    # Split the data into inputs and outputs
    inputs_preprocessed, outputs_preprocessed, scaler_x, scaler_y = preprocess_data(cleaned_combined)

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
