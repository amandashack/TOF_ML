import os
import io
import argparse
import h5py
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm, kurtosis, skew
from PyPDF2 import PdfReader, PdfWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
from loaders.load_and_save import DataGenerator, DataGeneratorWithVeto
from model_gen import y_tof_loss, time_of_flight_loss

def fig_to_pdf_page(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf')
    buf.seek(0)
    return PdfReader(buf).pages[0]


def load_scalers(scalers_path):
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers


def plot_histograms(data, scalers, sample_size=1000):
    # Randomly sample the data
    sampled_data = data[np.random.choice(data.shape[0], sample_size, replace=False), :]
    mask = sampled_data[:, 7].astype(bool)  # Use the mask to filter the data
    sampled_data = sampled_data[mask]

    # Calculate interaction terms and apply scaling
    generator = DataGenerator(sampled_data, scalers, batch_size=sample_size)
    data_with_interactions = generator.calculate_interactions(sampled_data[:, :5])
    scaled_data = generator.scale_input(data_with_interactions)

    # Create DataFrames for easier manipulation and plotting
    columns = [
        'initial_ke', 'elevation', 'retardation', 'mid1', 'mid2',
        'initial_ke*elevation', 'initial_ke*mid1', 'initial_ke*mid2',
        'elevation*retardation', 'elevation*mid1', 'elevation*mid2',
        'retardation*mid1', 'retardation*mid2',
        'initial_ke^2', 'elevation^2', 'retardation^2', 'mid1^2', 'mid2^2'
    ]
    df_unscaled = pd.DataFrame(data_with_interactions, columns=columns)
    df_scaled = pd.DataFrame(scaled_data, columns=columns)

    # Plot histograms for each column
    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        sns.histplot(df_unscaled[col], bins=30, kde=True, color='skyblue', edgecolor='k', ax=axes[0])
        axes[0].set_title(f'Unscaled {col}')
        sns.histplot(df_scaled[col], bins=30, kde=True, color='salmon', edgecolor='k', ax=axes[1])
        axes[1].set_title(f'Scaled {col}')
        plt.tight_layout()
        plt.show()

def load_test_data(test_data_path):
    with h5py.File(test_data_path, 'r') as hf:
        test_data = hf['test_data'][:]
    return test_data


# Function to randomly sample data
def random_sample_data(data, sample_size):
    indices = np.random.choice(data.shape[0], min(sample_size, data.shape[0]), replace=False)
    return data[indices]


def check_and_clean_data(data):
    # Replace infinities and NaNs with a large negative value
    data = np.nan_to_num(data, nan=-1e10, posinf=1e10, neginf=-1e10)
    return data

def plot_results(base_dir, model_num, fold, pdf_filename=None, sample_size=10000):
    veto_model_path = os.path.join(base_dir, str(model_num), f'veto_model_fold_{fold}.h5')
    main_model_path = os.path.join(base_dir, str(model_num), f'main_model_fold_{fold}.h5')
    veto_model = tf.keras.models.load_model(veto_model_path)
    main_model = tf.keras.models.load_model(main_model_path, custom_objects={'time_of_flight_loss': time_of_flight_loss,
                                                                             'y_tof_loss': y_tof_loss})

    test_data_path = os.path.join(base_dir, str(model_num), 'test_data.h5')
    test_data = load_test_data(test_data_path)

    # Adjust sample size if necessary
    test_data_ds = random_sample_data(test_data, sample_size)

    # Load scalers
    scalers_path = os.path.join(base_dir, str(model_num), 'scalers.pkl')
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    # Prepare test data
    generator = DataGenerator(test_data_ds, scalers)
    x_test = test_data_ds[:, :5]
    y_test = test_data_ds[:, 5:7]  # time_of_flight and y_pos
    data_with_interactions = generator.calculate_interactions(x_test)
    all_data = np.column_stack([data_with_interactions, y_test])
    data_scaled = generator.scale_input(all_data)

    input_batch = np.nan_to_num(data_scaled[:, :-2], nan=np.log(1e-10))
    output_batch = np.nan_to_num(data_scaled[:, -2:], nan=np.log(1e-10))

    # Generate the mask using the veto model
    mask = veto_model.predict(input_batch, verbose=0) > 0.5
    mask = mask.flatten().astype(bool)
    mask_true = test_data_ds[:, -1].astype(bool)
    matching_count = sum(1 for x,y in zip(mask, mask_true) if x==y)
    total_count = len(mask)
    print("Accuracy of veto model: ", matching_count/total_count)

    y_pred = main_model.predict(input_batch)
    y_pred_tof = y_pred[0].flatten()
    y_pred_y_pos = y_pred[1].flatten()

    y_pred_stack = np.column_stack([y_pred_tof, y_pred_y_pos])
    y_pred_inv = generator.inverse_scale_output(np.column_stack([input_batch, y_pred_stack]))
    y_true_inv = generator.inverse_scale_output(np.column_stack([input_batch, output_batch]))
    #print(y_pred_inv[:10, :], y_pred_inv[:10, -2:])

    # Convert to DataFrame
    df_tof = pd.DataFrame({
        'y_tof_pred': y_pred_inv[mask_true, -2],
        'y_pos_pred': y_pred_inv[mask_true, -1],
        'y_tof_test': y_true_inv[mask_true, -2].flatten(),
        'y_pos_test': y_true_inv[mask_true, -1].flatten(),
        'tof_residuals': y_true_inv[mask_true, -2].flatten() - y_pred_inv[mask_true, -2],
        'y_pos_residuals': y_true_inv[mask_true, -1].flatten() - y_pred_inv[mask_true, -1],
        'retardation': test_data_ds[mask_true, 2].flatten()
    })

    # Prepare to save plots to PDF if requested
    if pdf_filename:
        pdf_path = os.path.join(base_dir, str(model_num), pdf_filename)
        pdf_writer = PdfPages(pdf_path)

    # Scatter plot with regression line for time of flight and y_position
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, (col_pred, col_test, title) in enumerate(
            zip(['y_tof_pred', 'y_pos_pred'], ['y_tof_test', 'y_pos_test'], ['Time of Flight', 'Y Position'])):
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
    parser.add_argument('--sample_size', type=int, default=6000, help='Number of random samples to plot.')

    args = parser.parse_args()
    plot_results(args.base_dir, args.model_num, 2, pdf_filename=args.pdf_filename,
                 sample_size=args.sample_size)

if __name__ == '__main__':
    main()
