from loaders import MRCOLoader
import os
from sklearn.model_selection import train_test_split
import numpy as np
from model_gen import y0_NM
import h5py
import plotter
import matplotlib.pyplot as plt


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


def check_rebalance(data):
    hb = data>7.59
    high_bin= np.count_nonzero(hb)
    mb = np.logical_and(data>=4.96, data<7.59)
    mid_bin = np.count_nonzero(mb)
    lb = data<4.96
    low_bin = np.count_nonzero(lb)
    print(high_bin, mid_bin, low_bin, high_bin + mid_bin + low_bin)


def run_datasplit():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    amanda_filepath = dir_path + "\\NM_simulations"
    print(amanda_filepath)
    multi_retardation_sim = MRCOLoader(amanda_filepath)
    multi_retardation_sim.load()
    multi_retardation_sim.create_mask((402, np.inf), (0, 17.7), 5, "make it")
    training, val, test = multi_retardation_sim.rebalance(0.25, 3)
    x_train = training[0][:, 0:3]
    y_train = training[0][:, 3].T - y0_NM(training[0][:, 1].T)
    x_val = val[0][:, 0:3]
    y_val = val[0][:, 3].T - y0_NM(val[0][:, 1].T)
    x_test = test[0][:, 0:3]
    y_test = test[0][:, 3].T - y0_NM(test[0][:, 1].T)
    for i in range(1, 3):
        x_train = np.append(x_train, training[i][:, 0:3], axis=0)
        y_train = np.append(y_train, training[i][:, 3].T - y0_NM(training[i][:, 1].T), axis=0)
        x_val = np.append(x_val, val[i][:, 0:3], axis=0)
        y_val = np.append(y_val, val[i][:, 3].T - y0_NM(val[i][:, 1].T), axis=0)
        x_test = np.append(x_test, test[i][:, 0:3], axis=0)
        y_test = np.append(y_test, test[i][:, 3].T - y0_NM(test[i][:, 1].T), axis=0)
    train_dir_path = dir_path + "\\NM_simulations\\masked_data3\\train"
    test_dir_path = dir_path + "\\NM_simulations\\masked_data3\\test"
    validate_dir_path = dir_path + "\\NM_simulations\\masked_data3\\validate"
    generate_files(x_train, y_train, train_dir_path, "train_data.h5")
    generate_files(x_test, y_test, test_dir_path, "test_data.h5")
    generate_files(x_val, y_val, validate_dir_path, "validate_data.h5")


def multi_scatter(df):
    fig, ax = plt.subplots()
    ax.scatter(df[0], df[1])
    fig.tight_layout()
    plt.legend(prop={'size': 6})
    plt.show()


def plot_rebalanced_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    amanda_filepath = dir_path + "\\NM_simulations"
    print(amanda_filepath)
    multi_retardation_sim = MRCOLoader(amanda_filepath)
    multi_retardation_sim.load()
    multi_retardation_sim.create_mask((402, np.inf), (0, 17.7), 5, "make it")
    multi_retardation_sim.rebalance()
    #residual = multi_retardation_sim.data_masked[3, :] - y0_NM(multi_retardation_sim.data_masked[1, :])
    #X = multi_retardation_sim.data_masked[1, :]
    #Y = residual
    #fig, ax = plt.subplots()
    #plotter.one_plot_multi_scatter(ax, multi_retardation_sim.spec_masked, "Masked Data",
    #                               '$log_{2}$(Pass Energy)', "$log_{2}(TOF)$",
    #                               logarithm=True, fit=True)
    #fig.tight_layout()
    #plt.legend(prop={'size': 6})
    #plt.show()
    #multi_scatter([X, Y])

if __name__ == '__main__':
    run_datasplit()