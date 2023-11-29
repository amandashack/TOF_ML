from model_eval import evaluate
from loaders import train_test_val_loader
import sys
import os


def run_eval(params, model_path):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    train_filepath = dir_path + "\\NM_simulations\\masked_data1\\train"
    train_data = train_test_val_loader(train_filepath)
    val_filepath = dir_path + "\\NM_simulations\\masked_data1\\validate"
    val_data = train_test_val_loader(val_filepath)
    test_filepath = dir_path + "\\NM_simulations\\masked_data1\\test"
    test_data = train_test_val_loader(test_filepath)
    results = evaluate(model_path, train_data[:-1, :].T, train_data[-1, :],
                       val_data[:-1, :].T, val_data[-1, :],
                       test_data[:-1, :].T, test_data[-1, :],
                       params)
    print(results)


if __name__ == '__main__':
    params = sys.argv[1]
    params = {
        "dropout": 0.2,
        "layer_size": 6,
        "alpha": 0.001,
        "batch_size": 128,
        "epochs": 5
    }
    model_file_path = sys.argv[2]
    run_eval(params, model_file_path)