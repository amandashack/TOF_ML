from loaders import MRCOLoader
import os
from sklearn.model_selection import train_test_split
import numpy as np
from model_gen import y0_NM
import h5py


def generate_files(x_data, y_data, output_dir, filename):
    output_file_path = os.path.join(output_dir, filename)
    with h5py.File(output_file_path, "w") as f:
        g1 = f.create_group("data1")
        g1.create_dataset("elevation", data=x_data[:, 0].flatten())
        g1.create_dataset("pass", data=x_data[:, 1].flatten())
        g1.create_dataset("retardation", data=x_data[:, 2].flatten())
        g1.create_dataset("ele*ret", data=(x_data[:, 0].flatten() * x_data[:, 2].flatten()))
        g1.create_dataset("ele*pass", data=(x_data[:, 0].flatten() * x_data[:, 1].flatten()))
        g1.create_dataset("pass*ret", data=(x_data[:, 1].flatten() * x_data[:, 2].flatten()))
        g1.create_dataset("residuals", data=y_data[:])
        print("Data exported to:", output_file_path)


def run_datasplit():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    amanda_filepath = dir_path + "\\NM_simulations"
    print(amanda_filepath)
    multi_retardation_sim = MRCOLoader(amanda_filepath)
    multi_retardation_sim.load()
    multi_retardation_sim.create_mask((402, np.inf), (0, 17.7), "make it")
    residual = multi_retardation_sim.data_masked[3, :] - y0_NM(multi_retardation_sim.data_masked[1, :])
    X = multi_retardation_sim.data_masked[0:3, :]
    Y = residual
    # Split the model_data into train and test data
    # Separate the test data
    x, x_test, y, y_test = train_test_split(X.T, Y, test_size=0.15, shuffle=True)
    # Split the remaining data to train and validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, shuffle=True)
    train_dir_path = dir_path + "\\NM_simulations\\masked_data1\\train"
    test_dir_path = dir_path + "\\NM_simulations\\masked_data1\\test"
    validate_dir_path = dir_path + "\\NM_simulations\\masked_data1\\validate"
    generate_files(x_train, y_train, train_dir_path, "train_data.h5")
    generate_files(x_test, y_test, test_dir_path, "test_data.h5")
    generate_files(x_val, y_val, validate_dir_path, "validate_data.h5")


if __name__ == '__main__':
    run_datasplit()