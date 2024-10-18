import os
import argparse
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from models.tof_to_energy_model import TofToEnergyModel, InteractionLayer, ScalingLayer, LogTransformLayer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import kurtosis, skew, norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def load_test_data(data_filepath, test_indices):
    with h5py.File(data_filepath, 'r') as hf:
        sorted_indices = sorted(test_indices)
        data_rows = hf['combined_data'][sorted_indices]
    if data_rows.ndim == 1:
        data_rows = np.expand_dims(data_rows, axis=0)
    mask = data_rows[:, -1].astype(bool)
    masked_data_rows = data_rows[mask]
    retardation = masked_data_rows[:, 2]
    mask_retardation = retardation <= 0
    masked_data_rows = masked_data_rows[mask_retardation]
    if masked_data_rows.shape[0] == 0:
        # Return empty arrays if no data after filtering
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32)
        )
    input_data = masked_data_rows[:, [2, 3, 4, 5]].astype(np.float32)
    initial_ke = masked_data_rows[:, 0]
    pass_energy = initial_ke + masked_data_rows[:, 2]
    output_data = np.log2(pass_energy).reshape(-1, 1).astype(np.float32)
    return input_data, output_data


def format_model_params(params_dict):
    """
    Formats the model parameters dictionary into a readable string.
    """
    formatted_params = "\n".join([f"{key}: {value}" for key, value in params_dict.items()])
    return formatted_params


def parse_params_line(params_line):
    """
    Parses a params line into a dictionary.
    Example input: '--batch_size=1024 --dropout=0.2 --layer_size=32 --learning_rate=0.4 --optimizer=Adam'
    Returns: {'batch_size': '1024', 'dropout': '0.2', 'layer_size': '32', 'learning_rate': '0.4', 'optimizer': 'Adam'}
    """
    params_dict = {}
    if params_line:
        params_list = params_line.strip().split()
        for param in params_list:
            if param.startswith('--'):
                key_value = param[2:].split('=')
                if len(key_value) == 2:
                    key, value = key_value
                    params_dict[key] = value
    return params_dict

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


def plot_model_results(base_dir, model_dir_name, model_type, data_filepath, params_dict=None, pdf_filename=None, sample_size=1000):
    # Load the model
    model_path = os.path.join(base_dir, model_dir_name, 'main_model')
    model_type_display = model_type.replace('_', ' ')
    print(model_path, '\n\n\n')

    main_model = tf.keras.models.load_model(model_path, custom_objects={
        'LogTransformLayer': LogTransformLayer,
        'InteractionLayer': InteractionLayer,
        'ScalingLayer': ScalingLayer,
        'TofToEnergyModel': TofToEnergyModel
    })

    # Load scalers
    main_model.min_values, main_model.max_values = load_scalers(os.path.join(base_dir, model_dir_name, "scalers.pkl"))
    print("Model loaded with min_values:", main_model.min_values)
    print("Model loaded with max_values:", main_model.max_values)
    main_model.params = params_dict

    # Get model parameters
    if hasattr(main_model, 'params') and main_model.params:
        params = main_model.params
    else:
        params = params_dict or {}

    params_str = format_model_params(params)

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
        pdf_path = os.path.join('/sdf/home/a/ajshack/TOF_ML/figures', pdf_filename)
        pdf_writer = PdfPages(pdf_path)
        print(f"Plots will be saved to {pdf_path}")

    # --- Plot 1: Scatter Plot with Regression Line for Energy ---
    fig, ax = plt.subplots(figsize=(10, 6))

    mae = mean_absolute_error(df_tof['energy_true'], df_tof['energy_pred'])
    rmse = np.sqrt(mean_squared_error(df_tof['energy_true'], df_tof['energy_pred']))
    r_squared = r2_score(df_tof['energy_true'], df_tof['energy_pred'])

    scatter = sns.scatterplot(
        x='energy_pred',
        y='energy_true',
        data=df_tof,
        hue='retardation',
        palette='viridis',
        alpha=0.7,
        edgecolor='k',
        linewidth=0.5,
        s=80,
        ax=ax
    )
    sns.regplot(
        x='energy_pred',
        y='energy_true',
        data=df_tof,
        ci=95,
        scatter=False,
        line_kws={"color": "red"},
        ax=ax
    )
    ax.set_xlabel('Predicted Energy', fontsize=16)
    ax.set_ylabel('True Energy', fontsize=16)
    ax.set_title(f'Energy Performance ({model_type_display})', fontsize=18)

    # Inset for metrics and model parameters
    axins = inset_axes(ax, width="40%", height="30%", loc='upper left', borderpad=2)
    metrics_text = f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\n$R^2$: {r_squared:.3f}"
    model_text = f"Model Parameters:\n{params_str}"
    axins.text(0.01, 0.99, metrics_text, transform=axins.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    axins.text(0.01, 0.50, model_text, transform=axins.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    axins.axis('off')

    plt.tight_layout()
    if pdf_filename:
        pdf_writer.savefig(fig)
    else:
        plt.show()
    plt.close(fig)

    # --- Plot 2: Residual Plot for Energy ---
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        x='energy_pred',
        y='energy_residuals',
        data=df_tof,
        hue='retardation',
        palette='viridis',
        alpha=0.7,
        edgecolor='k',
        linewidth=0.5,
        s=80,
        ax=ax
    )
    ax.set_xlabel('Predicted Energy', fontsize=16)
    ax.set_ylabel('Residuals', fontsize=16)
    ax.set_title(f'Energy Residuals ({model_type_display})', fontsize=18)

    plt.tight_layout()
    if pdf_filename:
        pdf_writer.savefig(fig)
    else:
        plt.show()
    plt.close(fig)

    # --- Plot 3: Histogram of Residuals for Energy ---
    fig, ax = plt.subplots(figsize=(10, 6))

    residual_kurtosis = kurtosis(df_tof['energy_residuals'])
    residual_skewness = skew(df_tof['energy_residuals'])
    sns.histplot(
        df_tof['energy_residuals'],
        bins=30,
        kde=True,
        color='skyblue',
        edgecolor='k',
        linewidth=0.5,
        ax=ax
    )
    ax.set_xlabel('Residuals', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_title(f'Energy Residuals Distribution ({model_type_display})', fontsize=18)

    # Fit a normal distribution to residuals
    mu, std = norm.fit(df_tof['energy_residuals'])
    x = np.linspace(df_tof['energy_residuals'].min(), df_tof['energy_residuals'].max(), 100)
    pdf = norm.pdf(x, mu, std)

    # Scale the PDF to match the scale of the histogram
    bin_width = (df_tof['energy_residuals'].max() - df_tof['energy_residuals'].min()) / 30
    scaled_pdf = pdf * len(df_tof['energy_residuals']) * bin_width
    ax.plot(x, scaled_pdf, 'r-', lw=2, label='Fitted Normal Distribution')

    # Calculate FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * std

    # Inset for kurtosis, skewness, center, FWHM
    axins_hist = inset_axes(ax, width="40%", height="30%", loc='upper right', borderpad=2)
    axins_hist.text(0.01, 0.95, f'Kurtosis: {residual_kurtosis:.2f}', transform=axins_hist.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    axins_hist.text(0.01, 0.75, f'Skewness: {residual_skewness:.2f}', transform=axins_hist.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    axins_hist.text(0.01, 0.55, f'Center (μ): {mu:.2f}', transform=axins_hist.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    axins_hist.text(0.01, 0.35, f'FWHM: {fwhm:.2f}', transform=axins_hist.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    axins_hist.legend(['Fitted Normal Distribution'], fontsize=10)
    axins_hist.axis('off')

    ax.legend()

    plt.tight_layout()
    if pdf_filename:
        pdf_writer.savefig(fig)
    else:
        plt.show()
    plt.close(fig)

    # Save PDF if requested
    if pdf_filename:
        pdf_writer.close()
        print(f"All plots saved to {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot results for a trained model.')
    parser.add_argument('base_dir', type=str, help='Base directory where all saved models are stored.')
    parser.add_argument('model_dir_name', type=str, help='Model directory name to load and plot results for.')
    parser.add_argument('model_type', type=str, help='Model type (e.g., default, tofs_simple_swish, etc.)')
    parser.add_argument('data_filepath', type=str, help='Path to the data h5 file.')
    parser.add_argument('--params_line', type=str, help='Model parameters in string format.')
    parser.add_argument('--pdf_filename', type=str, help='Optional PDF filename to save plots.')
    parser.add_argument('--sample_size', type=int, default=20000, help='Number of random samples to plot.')

    args = parser.parse_args()

    # Parse the params_line into a dictionary
    params_dict = parse_params_line(args.params_line)

    plot_model_results(
        args.base_dir,
        args.model_dir_name,
        args.model_type,
        args.data_filepath,
        params_dict=params_dict,
        pdf_filename=args.pdf_filename,
        sample_size=args.sample_size
    )


if __name__ == '__main__':
    main()
