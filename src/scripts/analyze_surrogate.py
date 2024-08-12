import os
import argparse
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm, kurtosis, skew
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
from analysis_functions import *
sys.path.insert(0, os.path.abspath('..'))
from loaders.load_and_save import DataGenerator, DataGeneratorWithVeto


def plot_surrogate_results(base_dir, model_num, fold, pdf_filename=None, sample_size=10000):
    main_model_path = os.path.join(base_dir, str(model_num), f'main_model_fold_{fold}.h5')
    veto_model_path = os.path.join(base_dir, str(model_num), f'veto_model_fold_{fold}.h5')
    main_model = tf.keras.models.load_model(main_model_path, compile=False)
    veto_model = tf.keras.models.load_model(veto_model_path)

    test_data_path = os.path.join(base_dir, str(model_num), 'test_data.h5')
    test_data = load_test_data(test_data_path)

    # Adjust sample size if necessary
    test_data_ds = random_sample_data(test_data, sample_size)

    # Load scalers
    scalers_path = os.path.join(base_dir, str(model_num), 'scalers.pkl')
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    # Prepare test data
    generator = DataGeneratorWithVeto(test_data_ds, scalers, veto_model)
    x_test = test_data_ds[:, :5]
    y_test = test_data_ds[:, 5:7]  # time_of_flight and y_pos
    mask_true = test_data_ds[:, 7:8].flatten()
    data_with_interactions = generator.calculate_interactions(x_test)
    all_data = np.column_stack([data_with_interactions, np.zeros_like(y_test)])
    data_scaled = generator.scale_input(all_data)

    input_batch = np.nan_to_num(data_scaled[:, :-2], nan=np.log(1e-10))

    # Generate the mask using the veto model
    mask_pred = veto_model.predict(input_batch, verbose=0) > 0.5
    mask_pred = mask_pred.flatten().astype(bool)

    matching_count = sum(1 for x, y in zip(mask_true, mask_pred) if x == y)
    total_count = len(mask_true)
    accuracy = matching_count / total_count

    print(f"Accuracy: {accuracy:.2f}")

    y_pred = np.squeeze(main_model.predict(input_batch[mask_pred])).T

    y_pred_inv = generator.inverse_scale_output(np.column_stack([input_batch[mask_pred], y_pred]))

    # Convert to DataFrame
    df_tof = pd.DataFrame({
        'y_tof_pred': y_pred_inv[:, -2].flatten(),
        'y_tof_true': y_test[mask_pred, 0],
        'y_pos_pred': y_pred_inv[:, -1].flatten(),
        'y_pos_true': y_test[mask_pred, 1],
        'tof_residuals': y_test[mask_pred, 0] - y_pred_inv[:, -2].flatten(),
        'y_pos_residuals': y_test[mask_pred, 1] - y_pred_inv[:, -1].flatten(),
        'retardation': test_data_ds[mask_pred, 2].flatten()
    })

    # Prepare to save plots to PDF if requested
    if pdf_filename:
        pdf_path = os.path.join(base_dir, str(model_num), pdf_filename)
        pdf_writer = PdfPages(pdf_path)

    # Scatter plot with regression line for time of flight and y_position
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, (col_pred, col_test, title) in enumerate(
            zip(['y_tof_pred', 'y_pos_pred'], ['y_tof_true', 'y_pos_true'], ['Time of Flight', 'Y Position'])):
        print(col_pred, col_test)
        mae = mean_absolute_error(df_tof[col_test], df_tof[col_pred])
        rmse = np.sqrt(mean_squared_error(df_tof[col_test], df_tof[col_pred]))
        r_squared = r2_score(df_tof[col_test], df_tof[col_pred])

        sns.scatterplot(x=col_pred, y=col_test, data=df_tof, hue='retardation', palette='viridis', alpha=0.7,
                        edgecolor='k', linewidth=0.5, s=80, ax=axs[i])
        sns.regplot(x=col_pred, y=col_test, data=df_tof, ci=95, scatter_kws={'alpha': 0}, line_kws={"color": "red"},
                    ax=axs[i])
        axs[i].set_xlabel('Predicted Values', fontsize=16)
        axs[i].set_ylabel('True Values', fontsize=16)
        axs[i].set_title(f'{title} Performance')

        # Inset for metrics
        axins = inset_axes(axs[i], width="40%", height="30%", loc='upper right')
        axins.text(0.05, 0.7, f'MAE = {mae:.3f}', fontsize=14)
        axins.text(0.05, 0.5, f'RMSE = {rmse:.3f}', fontsize=14)
        axins.text(0.05, 0.3, f'$R^2 = {r_squared:.3f}$', fontsize=14)
        axins.axis('off')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Residual plots for time of flight and y_position
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, (col_pred, col_res, title) in enumerate(
            zip(['y_tof_pred', 'y_pos_pred'], ['tof_residuals', 'y_pos_residuals'], ['Time of Flight', 'Y Position'])):
        sns.scatterplot(x=col_pred, y=col_res, data=df_tof, hue='retardation', palette='viridis', alpha=0.7,
                        edgecolor='k', linewidth=0.5, s=80, ax=axs[i])
        axs[i].set_xlabel('Predicted Values', fontsize=16)
        axs[i].set_ylabel('Residuals', fontsize=16)
        axs[i].set_title(f'{title} Residuals')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Histograms of residuals for time of flight and y_position
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, (col_res, title) in enumerate(zip(['tof_residuals', 'y_pos_residuals'], ['Time of Flight', 'Y Position'])):
        residual_kurtosis, residual_skewness = kurtosis(df_tof[col_res]), skew(df_tof[col_res])
        sns.histplot(df_tof[col_res], bins=30, kde=True, color='skyblue', edgecolor='k', linewidth=0.5, ax=axs[i])
        axs[i].set_xlabel('Residuals', fontsize=16)
        axs[i].set_ylabel('Frequency', fontsize=16)
        axs[i].set_title(f'{title} Residuals Distribution')

        # Fit a normal distribution to residuals
        mu, std = norm.fit(df_tof[col_res])
        x = np.linspace(min(df_tof[col_res]), max(df_tof[col_res]), 100)
        pdf = norm.pdf(x, mu, std)

        # Scale the PDF to match the scale of the histogram
        bin_width = (max(df_tof[col_res]) - min(df_tof[col_res])) / 30
        scaled_pdf = pdf * len(df_tof[col_res]) * bin_width
        axs[i].plot(x, scaled_pdf, 'r-', lw=2)
        axs[i].legend(['Fitted Normal Distribution'])

        # Calculate FWHM
        fwhm = 2 * np.sqrt(2 * np.log(2)) * std

        # Inset for kurtosis, skewness, center, and FWHM
        axins_hist = inset_axes(axs[i], width="40%", height="40%", loc='upper right')
        axins_hist.text(0.3, 0.7, f'Kurtosis = {residual_kurtosis:.2f}', fontsize=12)
        axins_hist.text(0.3, 0.5, f'Skewness = {residual_skewness:.2f}', fontsize=12)
        axins_hist.text(0.3, 0.3, f'Center = {mu:.2f}', fontsize=12)
        axins_hist.text(0.3, 0.1, f'FWHM = {fwhm:.2f}', fontsize=12)
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
    plot_surrogate_results(args.base_dir, args.model_num, 1, pdf_filename=args.pdf_filename,
                 sample_size=args.sample_size)

if __name__ == '__main__':
    main()
