import os
import tensorflow as tf
import numpy as np
import pickle
import argparse


def get_params_for_model(params_file, model_num):
    with open(params_file, 'r') as f:
        lines = f.readlines()
        params_line = lines[model_num - 1].strip()
        params_dict = dict(item.split('=') for item in params_line.split(' --'))
        return params_dict


def combine_folds(base_dir, model_num, n_folds=3):
    params_file = os.path.join(base_dir, 'params')
    params = get_params_for_model(params_file, model_num)

    # Load models from each fold
    fold_models = []
    for fold in range(1, n_folds + 1):
        model_path = os.path.join(base_dir, str(model_num), f'main_model_fold_{fold}.h5')
        model = tf.keras.models.load_model(model_path, compile=False)
        fold_models.append(model)

    # Create a new model for the combined weights
    combined_model = create_main_model(params, steps_per_execution=1)  # Adjust steps_per_execution if needed

    # Average the weights from all folds
    combined_weights = [model.get_weights() for model in fold_models]
    new_weights = []

    for weights_tuple in zip(*combined_weights):
        new_weights.append(np.mean(np.array(weights_tuple), axis=0))

    combined_model.set_weights(new_weights)

    # Save the combined model
    combined_model_path = os.path.join(base_dir, str(model_num), 'combined_model.h5')
    combined_model.save(combined_model_path)
    print(f"Combined model saved at {combined_model_path}")

    # Save the params to a file for later use in plotting
    with open(os.path.join(base_dir, str(model_num), 'combined_model_params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # Load test data
    test_data_path = os.path.join(base_dir, str(model_num), 'test_data.h5')
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    # Load scalers
    scalers_path = os.path.join(base_dir, str(model_num), 'scalers.pkl')
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    # Evaluate the combined model
    x_test, y_test = preprocess_surrogate_test_data(test_data, scalers, None, combined_model)
    loss_test = combined_model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test loss: {loss_test[0]:.4f}, Test MAE: {loss_test[1]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine model weights from different folds.')
    parser.add_argument('base_dir', type=str, help='Base directory where all saved models are stored.')
    parser.add_argument('model_num', type=int, help='Model number to combine weights for.')
    parser.add_argument('--n_folds', type=int, default=3, help='Number of folds to combine.')
    args = parser.parse_args()

    combine_folds(args.base_dir, args.model_num, args.n_folds)