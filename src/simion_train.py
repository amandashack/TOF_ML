"""
This file is a sandbox for testing/running functions
"""

from loaders import MRCOLoader, train_test_val_loader
from plotter import one_plot_multi_scatter, pass_versus_counts
import matplotlib.pyplot as plt
import numpy as np
from model_gen import run_model
import sys
import os
import tensorflow as tf
from model_eval import evaluate


def run_train(out_path, params):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    train_filepath = dir_path + "\\NM_simulations\\masked_data1\\train"
    train_data = train_test_val_loader(train_filepath)
    val_filepath = dir_path + "\\NM_simulations\\masked_data1\\validate"
    val_data = train_test_val_loader(val_filepath)
    test_filepath = dir_path + "\\NM_simulations\\masked_data1\\test"
    test_data = train_test_val_loader(test_filepath)
    model = run_model(train_data[:-1, :].T, train_data[-1, :], val_data[:-1, :].T, val_data[-1, :], params)
    model.save(out_path)

    loss_train = model.history['loss']
    loss_val = model.history['val_loss']
    loss_test = evaluate(mode, x_test, y_test)

    print(f"test_loss {loss_test}", f"train_loss {loss_train}", f"val_loss {loss_val}")


if __name__ == '__main__':
    params = sys.argv[2]
    print(params)
    output_file_path = sys.argv[1]
    print(out_file_path)
    run_train(output_file_path, params)
