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
import re


def run_train(out_path, params):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    train_filepath = dir_path + "/NM_simulations/masked_data1/train"
    train_data = train_test_val_loader(train_filepath)
    val_filepath = dir_path + "/NM_simulations/masked_data1/validate"
    val_data = train_test_val_loader(val_filepath)
    test_filepath = dir_path + "/NM_simulations/masked_data1/test"
    test_data = train_test_val_loader(test_filepath)
    model, history = run_model(train_data[:-1, :].T, train_data[-1, :], 
            val_data[:-1, :].T, val_data[-1, :], params)
    model.save(out_path)
    x_test = test_data[:-1, :].T
    y_test = test_data[-1, :]
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    loss_test = evaluate(model, x_test, y_test)

    print(f"test_loss {loss_test}")


if __name__ == '__main__':
    p = ' '.join(sys.argv[2:])
    p = re.findall(r'(\w+)=(\d+)', p)
    params = dict((p[i][0], float(p[i][1])) for i in range(len(p)))
    output_file_path = sys.argv[1]
    run_train(output_file_path, params)
