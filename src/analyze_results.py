import os
import io
import argparse
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm, kurtosis, skew
from PyPDF2 import PdfReader, PdfWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from model_gen import time_of_flight_loss, y_tof_loss, hit_loss
from matplotlib.backends.backend_pdf import PdfPages


def fig_to_pdf_page(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf')
    buf.seek(0)
    return PdfReader(buf).pages[0]


def load_test_data(test_data_path):
    with h5py.File(test_data_path, 'r') as hf:
        test_data = hf['test_data'][:]
    return test_data


# Function to randomly sample data
def random_sample_data(df, sample_size):
    return df.sample(n=min(sample_size, len(df)), random_state=42)


# Function to plot the results
def plot_results(base_dir, model_num, pdf_filename=None, sample_size=1000):
    model_path = os.path.join(base_dir, str(model_num))
    model = tf.keras.models.load_model(model_path, custom_objects={'time_of_flight_loss': time_of_flight_loss,
                                                                   'y_tof_loss': y_tof_loss, 'hit_loss': hit_loss})

    test_data_path = os.path.join(model_path, 'test_data.h5')
    test_data = load_test_data(test_data_path)
    retardation_values = test_data[:2*sample_size, 2].flatten()
    x_test = test_data[:2*sample_size, :5]
    print(x_test[:100, :])
    y_test = test_data[:2*sample_size, 5:]
    print(y_test[:100, :])
    y_pred = model.predict(x_test)
    print(y_pred[i][:100] for i in range(len(y_pred)))

    # Mask the data
    mask = y_test[:, 2].flatten().astype(bool)
    y_test = y_test[mask]
    y_pred = [y[mask] for y in y_pred]
    retardation_values = retardation_values[mask]

    tof_residuals = y_test[:, 0].flatten() - y_pred[0].flatten()
    y_residuals = y_test[:, 1].flatten() - y_pred[1].flatten()

    # Prepare data for plotting
    df_tof = pd.DataFrame({
        'y_pred': y_pred[0].flatten(),
        'y_test': y_test[:, 0].flatten(),
        'residuals': tof_residuals.flatten(),
        'retardation': retardation_values
    })

    df_y_tof = pd.DataFrame({
        'y_pred': y_pred[1].flatten(),
        'y_test': y_test[:, 1].flatten(),
        'residuals': y_residuals.flatten(),
        'retardation': retardation_values
    })

    # Adjust sample size if necessary
    df_tof = random_sample_data(df_tof, sample_size)
    df_y_tof = random_sample_data(df_y_tof, sample_size)

    # Prepare to save plots to PDF if requested
    if pdf_filename:
        pdf_writer = PdfPages(pdf_filename)

    # Scatter plot with regression line for time of flight
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, (df, title) in enumerate(zip([df_tof, df_y_tof], ['Time of Flight', 'Y TOF'])):
        mae = mean_absolute_error(df['y_test'], df['y_pred'])
        rmse = np.sqrt(mean_squared_error(df['y_test'], df['y_pred']))
        r_squared = r2_score(df['y_test'], df['y_pred'])

        sns.scatterplot(x='y_pred', y='y_test', data=df, hue='retardation', palette='viridis', alpha=0.7,
                        edgecolor='k', linewidth=0.5, s=80, ax=axs[i])
        sns.regplot(x='y_pred', y='y_test', data=df, ci=95, scatter_kws={'alpha': 0}, line_kws={"color": "red"},
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

    # Residual plot for time of flight and y tof
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, (df, title) in enumerate(zip([df_tof, df_y_tof], ['Time of Flight', 'Y TOF'])):
        sns.scatterplot(x='y_pred', y='residuals', data=df, hue='retardation', palette='viridis', alpha=0.7,
                        edgecolor='k', linewidth=0.5, s=80, ax=axs[i])
        axs[i].set_xlabel('Predicted Values', fontsize=16)
        axs[i].set_ylabel('Residuals', fontsize=16)
        axs[i].set_title(f'{title} Residuals')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Histogram of residuals for time of flight and y tof
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, (df, title) in enumerate(zip([df_tof, df_y_tof], ['Time of Flight', 'Y TOF'])):
        residual_kurtosis, residual_skewness = kurtosis(df['residuals']), skew(df['residuals'])
        sns.histplot(df['residuals'], bins=30, kde=True, color='skyblue', edgecolor='k', linewidth=0.5, ax=axs[i])
        axs[i].set_xlabel('Residuals', fontsize=16)
        axs[i].set_ylabel('Frequency', fontsize=16)
        axs[i].set_title(f'{title} Residuals Distribution')

        # Fit a normal distribution to residuals
        mu, std = norm.fit(df['residuals'])
        x = np.linspace(min(df['residuals']), max(df['residuals']), 100)
        pdf = norm.pdf(x, mu, std)

        # Scale the PDF to match the scale of the histogram
        bin_width = (max(df['residuals']) - min(df['residuals'])) / 30
        scaled_pdf = pdf * len(df['residuals']) * bin_width
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
    parser.add_argument('--sample_size', type=int, default=60000, help='Number of random samples to plot.')

    args = parser.parse_args()
    plot_results(args.base_dir, args.model_num, args.pdf_filename, args.sample_size)

if __name__ == '__main__':
    main()
