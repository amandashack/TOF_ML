import os
import numpy as np
import h5py
import tensorflow as tf
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew, norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse
from models.tof_to_energy_model import TofToEnergyModel


# Custom layers used in the model
class LogTransformLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LogTransformLayer, self).__init__(**kwargs)

    def call(self, inputs):
        log_transformed = tf.concat([
            inputs[:, 0:3],
            tf.math.log(inputs[:, 3:4]) / tf.math.log(2.0)
        ], axis=1)
        return log_transformed


class InteractionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InteractionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x1 = inputs[:, 0:1]
        x2 = inputs[:, 1:2]
        x3 = inputs[:, 2:3]
        x4 = inputs[:, 3:4]
        interaction_terms = tf.concat([
            x1 * x2,
            x1 * x3,
            x1 * x4,
            x2 * x3,
            x2 * x4,
            x3 * x4,
            tf.square(x4)
        ], axis=1)
        return tf.concat([inputs, interaction_terms], axis=1)


class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, min_values, max_values, **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)
        self.min_values = tf.constant(min_values, dtype=tf.float32)
        self.max_values = tf.constant(max_values, dtype=tf.float32)

    def call(self, inputs):
        return (inputs - self.min_values) / (self.max_values - self.min_values)


def process_input(input_data):
    x1 = input_data[:, 0:1]
    x2 = input_data[:, 1:2]
    x3 = input_data[:, 2:3]
    x4 = input_data[:, 3:4]
    interaction_terms = np.hstack([
        x1 * x2,
        x1 * x3,
        x1 * x4,
        x2 * x3,
        x2 * x4,
        x3 * x4,
        x4 ** 2
    ])
    return np.hstack([input_data, interaction_terms])


def load_test_data(data_filepath, test_indices):
    input_data_list = []
    output_data_list = []
    with h5py.File(data_filepath, 'r') as hf:
        print("opened h5 file and working on getting data")
        for idx in test_indices:
            data_row = hf['combined_data'][idx]
            if data_row.ndim == 1:
                data_row = np.expand_dims(data_row, axis=0)
            mask = data_row[:, -1].astype(bool)
            masked_data_row = data_row[mask]
            if masked_data_row.shape[0] == 0:
                continue
            input_data = masked_data_row[:, [2, 3, 4, 5]]
            output_data = masked_data_row[:, 0].reshape(-1, 1)
            output_data = np.log2(output_data)
            for inp, outp in zip(input_data, output_data):
                input_data_list.append(inp.astype(np.float32))
                output_data_list.append(outp.astype(np.float32))
    input_data_array = np.array(input_data_list)
    output_data_array = np.array(output_data_list)
    print("data set created")
    return input_data_array, output_data_array

def load_scalers(scalers_path):
    if os.path.exists(scalers_path):
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
            min_values = scalers['min_values']
            max_values = scalers['max_values']
            print(f"Scalers loaded from {scalers_path}")
            return min_values, max_values
    else:
        raise FileNotFoundError(f"Scalers file not found at {scalers_path}")


def plot_model_results(base_dir, model_dir_name, model_type, data_filepath, pdf_filename=None, sample_size=1000):
    # Load the model
    model_path = os.path.join(base_dir, model_dir_name, 'main_model')
    print(model_path, '\n\n\n')
    # i believe that from config and get config handle the scaling and param values
    main_model = tf.keras.models.load_model(model_path, custom_objects={
        'LogTransformLayer': LogTransformLayer,
        'InteractionLayer': InteractionLayer,
        'ScalingLayer': ScalingLayer,
        'TofToEnergyModel': TofToEnergyModel
    })
    #print(main_model.min_values, main_model.max_values, main_model.params)
    main_model.min_values, main_model.max_values = load_scalers(os.path.join(base_dir, model_dir_name, "scalers.pkl"))
    
    # Load test indices
    indices_path = os.path.join(base_dir, model_dir_name, 'data_indices.npz')
    indices_data = np.load(indices_path)
    test_indices = indices_data['test_indices']

    if sample_size:
        sample_indices = np.random.choice(test_indices, sample_size, replace=False)


    # Load test data
    input_data, output_data = load_test_data(data_filepath, sample_indices)

    # Adjust sample size if necessary
    #if sample_size and len(input_data) > sample_size:
    #    sample_indices = np.random.choice(len(input_data), sample_size, replace=False)
    #    input_data = input_data[sample_indices]
    #    output_data = output_data[sample_indices]

    # Get predictions
    y_pred = main_model.predict(input_data).flatten()

    # Invert transformations
    energy_pred = 2 ** y_pred
    energy_true = 2 ** output_data.flatten()
    retardation = input_data[:, 0]

    # Create DataFrame
    df_tof = pd.DataFrame({
        'energy_pred': energy_pred,
        'energy_true': energy_true,
        'energy_residuals': energy_true - energy_pred,
        'retardation': retardation.flatten()
    })

    # Prepare to save plots to PDF if requested
    if pdf_filename:
        pdf_path = os.path.join(base_dir, model_dir_name, pdf_filename)
        pdf_writer = PdfPages(pdf_path)

    # Plotting code remains the same as before, adjusted for new data structures
    # (Include scatter plots, residual plots, histograms, etc.)

    # Scatter plot with regression line for energy
    fig, axs = plt.subplots(figsize=(10, 6))

    mae = mean_absolute_error(df_tof['energy_true'], df_tof['energy_pred'])
    rmse = np.sqrt(mean_squared_error(df_tof['energy_true'], df_tof['energy_pred']))
    r_squared = r2_score(df_tof['energy_true'], df_tof['energy_pred'])

    sns.scatterplot(x='energy_pred', y='energy_true', data=df_tof, hue='retardation', palette='viridis', alpha=0.7,
                    edgecolor='k', linewidth=0.5, s=80, ax=axs)
    sns.regplot(x='energy_pred', y='energy_true', data=df_tof, ci=95, scatter_kws={'alpha': 0}, line_kws={"color": "red"},
                ax=axs)
    axs.set_xlabel('Predicted Energy', fontsize=16)
    axs.set_ylabel('True Energy', fontsize=16)
    axs.set_title(f'Energy Performance')

    # Inset for metrics
    axins = inset_axes(axs, width="50%", height="40%", loc='upper right')
    axins.text(0.05, 0.7, f'MAE = {mae:.3f}', fontsize=12)
    axins.text(0.05, 0.5, f'RMSE = {rmse:.3f}', fontsize=12)
    axins.text(0.05, 0.3, f'$R^2 = {r_squared:.3f}$', fontsize=12)
    axins.axis('off')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Additional plots (residual plots, histograms) go here

    # Save PDF if requested
    if pdf_filename:
        pdf_writer.close()

def main():
    parser = argparse.ArgumentParser(description='Plot results for a trained model.')
    parser.add_argument('base_dir', type=str, help='Base directory where all saved models are stored.')
    parser.add_argument('model_dir_name', type=str, help='Model directory name to load and plot results for.')
    parser.add_argument('model_type', type=str, help='Model type (e.g., default, tofs_simple_swish, etc.)')
    parser.add_argument('data_filepath', type=str, help='Path to the data h5 file.')
    parser.add_argument('--pdf_filename', type=str, help='Optional PDF filename to save plots.')
    parser.add_argument('--sample_size', type=int, default=20000, help='Number of random samples to plot.')

    args = parser.parse_args()
    plot_model_results(args.base_dir, args.model_dir_name, args.model_type, args.data_filepath,
                       pdf_filename=args.pdf_filename, sample_size=args.sample_size)

if __name__ == '__main__':
    main()
