import argparse
import os
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from loaders.model_data_generator import load_from_h5
from scripts.analysis_functions import random_sample_data
from loaders.model_data_generator import DataGeneratorTofToKE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm, kurtosis, skew
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


def preprocess_test_data(main_model, input_data, output_data, scalers):
    y_pred = np.squeeze(main_model.predict(input_data)).T
    y_pred_inv = scalers.inverse_transform(np.column_stack([y_pred, input_data]))

    # Convert to DataFrame
    df_tof = pd.DataFrame({
        'energy_pred': 2**(y_pred_inv[:, 0].flatten()),
        'energy_true': 2**(output_data.flatten()),
        'energy_residuals': 2**(output_data.flatten()) - 2**(y_pred_inv[:, 0].flatten()),
        'retardation': y_pred_inv[:, 1].flatten()
    })
    return df_tof

def plot_model_results(base_dir, model_num, pdf_filename=None, sample_size=20000):
    model_path = os.path.join(base_dir, str(model_num), 'main_model.h5')

    main_model = tf.keras.models.load_model(model_path, compile=False)

    test_data_path = os.path.join(base_dir, str(model_num), 'test_data.h5')
    test_data = load_from_h5(test_data_path, grp='test_data')
    mask = test_data[:, -1].astype(bool)
    test_data_masked = test_data[mask]

    # Adjust sample size if necessary
    test_data_ds = random_sample_data(test_data_masked, sample_size)

    # Load scalers
    scalers_path = os.path.join(base_dir, str(model_num), 'scalers.pkl')
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    epsilon = 1e-10
    test_data_ds[:, 0] = np.log2(test_data_ds[:, 0] + epsilon)
    test_data_ds[:, 5] = np.log2(test_data_ds[:, 5] + epsilon)

    # Extract input and output data
    input_data = test_data_ds[:, [2, 3, 4, 5]]
    output_data = test_data_ds[:, 0].reshape(-1, 1)
    input_data = DataGeneratorTofToKE.process_input(input_data)

    # Stack output and input data
    full_batch = np.hstack([output_data.copy(), input_data.copy()])

    # Apply scaling
    scaled_full_batch = scalers.transform(full_batch)
    scaled_input_data = scaled_full_batch[:, 1:]

    df_tof = preprocess_test_data(main_model, scaled_input_data, output_data, scalers)

    # TODO: make this an option either through inputting params, a params file, or without it just skip
    # Load the params used for this model
    # with open(os.path.join(base_dir, str(model_num), 'combined_model_params.pkl'), 'rb') as f:
    #    params = pickle.load(f)

    # Convert params to string for display
    # params_str = ', '.join([f"{key}={value}" for key, value in params.items()])

    # Prepare to save plots to PDF if requested
    if pdf_filename:
        pdf_path = os.path.join(base_dir, str(model_num), pdf_filename)
        pdf_writer = PdfPages(pdf_path)

    # Scatter plot with regression line for time of flight and y_position
    fig, axs = plt.subplots(figsize=(10, 6))

    for i, (col_pred, col_test, title) in enumerate(
            zip(['energy_pred'], ['energy_true'], ['Energy'])):
        mae = mean_absolute_error(df_tof['energy_true'], df_tof['energy_pred'])
        rmse = np.sqrt(mean_squared_error(df_tof['energy_true'], df_tof['energy_pred']))
        r_squared = r2_score(df_tof['energy_true'], df_tof['energy_pred'])

        sns.scatterplot(x=col_pred, y=col_test, data=df_tof, hue='retardation', palette='viridis', alpha=0.7,
                        edgecolor='k', linewidth=0.5, s=80, ax=axs)
        sns.regplot(x=col_pred, y=col_test, data=df_tof, ci=95, scatter_kws={'alpha': 0}, line_kws={"color": "red"},
                    ax=axs)
        axs.set_xlabel('Predicted Values', fontsize=16)
        axs.set_ylabel('True Values', fontsize=16)
        axs.set_title(f'{title} Performance')

        # Inset for metrics and params
        axins = inset_axes(axs, width="50%", height="40%", loc='upper right')
        axins.text(0.05, 0.7, f'MAE = {mae:.3f}', fontsize=12)
        axins.text(0.05, 0.5, f'RMSE = {rmse:.3f}', fontsize=12)
        axins.text(0.05, 0.3, f'$R^2 = {r_squared:.3f}$', fontsize=12)
        # axins.text(0.05, 0.1, params_str, fontsize=10)
        axins.axis('off')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Residual plots for time of flight and y_position
    fig, axs = plt.subplots(1, 1, figsize=(16, 6))

    for i, (col_pred, col_res, title) in enumerate(
            zip(['energy_pred'], ['energy_true'], ['Energy'])):
        sns.scatterplot(x=col_pred, y=col_res, data=df_tof, hue='retardation', palette='viridis', alpha=0.7,
                        edgecolor='k', linewidth=0.5, s=80, ax=axs)
        axs.set_xlabel('Predicted Values', fontsize=16)
        axs.set_ylabel('Residuals', fontsize=16)
        axs.set_title(f'{title} Residuals')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Histograms of residuals for time of flight and y_position
    fig, axs = plt.subplots(1, 1, figsize=(16, 6))

    for i, (col_res, title) in enumerate(zip(['energy_residuals'], ['Energy'])):
        residual_kurtosis, residual_skewness = kurtosis(df_tof[col_res]), skew(df_tof[col_res])
        sns.histplot(df_tof[col_res], bins=30, kde=True, color='skyblue', edgecolor='k', linewidth=0.5, ax=axs[i])
        axs.set_xlabel('Residuals', fontsize=16)
        axs.set_ylabel('Frequency', fontsize=16)
        axs.set_title(f'{title} Residuals Distribution')

        # Fit a normal distribution to residuals
        mu, std = norm.fit(df_tof[col_res])
        x = np.linspace(min(df_tof[col_res]), max(df_tof[col_res]), 100)
        pdf = norm.pdf(x, mu, std)

        # Scale the PDF to match the scale of the histogram
        bin_width = (max(df_tof[col_res]) - min(df_tof[col_res])) / 30
        scaled_pdf = pdf * len(df_tof[col_res]) * bin_width
        axs.plot(x, scaled_pdf, 'r-', lw=2)
        axs.legend(['Fitted Normal Distribution'])

        # Calculate FWHM
        fwhm = 2 * np.sqrt(2 * np.log(2)) * std

        # Inset for kurtosis, skewness, center, FWHM, and params
        axins_hist = inset_axes(axs, width="50%", height="40%", loc='upper right')
        axins_hist.text(0.3, 0.7, f'Kurtosis = {residual_kurtosis:.2f}', fontsize=12)
        axins_hist.text(0.3, 0.5, f'Skewness = {residual_skewness:.2f}', fontsize=12)
        axins_hist.text(0.3, 0.3, f'Center = {mu:.2f}', fontsize=12)
        axins_hist.text(0.3, 0.1, f'FWHM = {fwhm:.2f}', fontsize=12)
        #axins_hist.text(0.05, 0.1, params_str, fontsize=10)
        axins_hist.axis('off')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Save PDF if requested
    if pdf_filename:
        pdf_writer.close()


def main():
    parser = argparse.ArgumentParser(description='Plot results for a trained model.')
    parser.add_argument('base_dir', type=str, help='Base directory where all saved models are stored.')
    parser.add_argument('model_num', type=int, help='Model number to load and plot results for.')
    parser.add_argument('--pdf_filename', type=str, help='Optional PDF filename to save plots.')
    parser.add_argument('--sample_size', type=int, default=20000, help='Number of random samples to plot.')

    args = parser.parse_args()
    plot_model_results(args.base_dir, args.model_num, pdf_filename=args.pdf_filename,
                           sample_size=args.sample_size)


if __name__ == '__main__':
    main()
