import os
import io
import argparse
import h5py
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm, kurtosis, skew
from PyPDF2 import PdfReader, PdfWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from model_gen import time_of_flight_loss
from matplotlib.backends.backend_pdf import PdfPages
from loaders.load_and_save import DataGenerator

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
    model = tf.keras.models.load_model(model_path, custom_objects={'time_of_flight_loss': time_of_flight_loss})

    test_data_path = os.path.join(model_path, 'test_data.h5')
    test_data = load_test_data(test_data_path)

    # Load scalers
    scalers_path = os.path.join(model_path, 'scalers.pkl')
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    # Prepare test data
    generator = DataGenerator(test_data, scalers, batch_size=len(test_data))
    x_test = generator.calculate_interactions(test_data[:, :5])
    y_test = test_data[:, 5:6]  # time_of_flight
    y_test = np.where(y_test <= 0, 1e-10, y_test)
    y_test = np.log(y_test)

    # Mask and predict
    mask = test_data[:, 7].astype(bool)
    x_test = x_test[mask]
    y_test = y_test[mask]

    y_pred = model.predict(x_test)
    y_pred = generator.inverse_scale_output(y_pred)
    y_test = generator.inverse_scale_output(y_test)

    # Convert to DataFrame
    df_tof = pd.DataFrame({
        'y_pred': y_pred.flatten(),
        'y_test': y_test.flatten(),
        'residuals': y_test.flatten() - y_pred.flatten(),
        'retardation': test_data[:, 2].flatten()[mask]
    })

    # Adjust sample size if necessary
    df_tof = random_sample_data(df_tof, sample_size)

    # Prepare to save plots to PDF if requested
    if pdf_filename:
        pdf_writer = PdfPages(pdf_filename)

    # Scatter plot with regression line for time of flight
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    mae = mean_absolute_error(df_tof['y_test'], df_tof['y_pred'])
    rmse = np.sqrt(mean_squared_error(df_tof['y_test'], df_tof['y_pred']))
    r_squared = r2_score(df_tof['y_test'], df_tof['y_pred'])

    sns.scatterplot(x='y_pred', y='y_test', data=df_tof, hue='retardation', palette='viridis', alpha=0.7,
                    edgecolor='k', linewidth=0.5, s=80, ax=axs)
    sns.regplot(x='y_pred', y='y_test', data=df_tof, ci=95, scatter_kws={'alpha': 0}, line_kws={"color": "red"},
                ax=axs)
    axs.set_xlabel('Predicted Values', fontsize=16)
    axs.set_ylabel('True Values', fontsize=16)
    axs.set_title('Time of Flight Performance')

    # Inset for metrics
    axins = inset_axes(axs, width="40%", height="30%", loc='upper right')
    axins.text(0.05, 0.7, f'MAE = {mae:.3f}', fontsize=14)
    axins.text(0.05, 0.5, f'RMSE = {rmse:.3f}', fontsize=14)
    axins.text(0.05, 0.3, f'$R^2 = {r_squared:.3f}$', fontsize=14)
    axins.axis('off')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Residual plot for time of flight
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    sns.scatterplot(x='y_pred', y='residuals', data=df_tof, hue='retardation', palette='viridis', alpha=0.7,
                    edgecolor='k', linewidth=0.5, s=80, ax=axs)
    axs.set_xlabel('Predicted Values', fontsize=16)
    axs.set_ylabel('Residuals', fontsize=16)
    axs.set_title('Time of Flight Residuals')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Histogram of residuals for time of flight
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    residual_kurtosis, residual_skewness = kurtosis(df_tof['residuals']), skew(df_tof['residuals'])
    sns.histplot(df_tof['residuals'], bins=30, kde=True, color='skyblue', edgecolor='k', linewidth=0.5, ax=axs)
    axs.set_xlabel('Residuals', fontsize=16)
    axs.set_ylabel('Frequency', fontsize=16)
    axs.set_title('Time of Flight Residuals Distribution')

    # Fit a normal distribution to residuals
    mu, std = norm.fit(df_tof['residuals'])
    x = np.linspace(min(df_tof['residuals']), max(df_tof['residuals']), 100)
    pdf = norm.pdf(x, mu, std)

    # Scale the PDF to match the scale of the histogram
    bin_width = (max(df_tof['residuals']) - min(df_tof['residuals'])) / 30
    scaled_pdf = pdf * len(df_tof['residuals']) * bin_width
    axs.plot(x, scaled_pdf, 'r-', lw=2)
    axs.legend(['Fitted Normal Distribution'])

    # Calculate FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * std

    # Inset for kurtosis, skewness, center, and FWHM
    axins_hist = inset_axes(axs, width="40%", height="40%", loc='upper right')
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
