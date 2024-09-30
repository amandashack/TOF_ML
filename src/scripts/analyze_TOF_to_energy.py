import tensorflow as tf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from analysis_functions import *
from scipy.stats import norm, kurtosis, skew
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import matplotlib
import argparse
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add parent directory to system path
sys.path.insert(0, parent_dir)
from loaders.model_data_generator import DataGeneratorTofToKE

def plot_TOF_to_energy_results(base_dir, model_num, fold, pdf_filename=None, sample_size=10000):
    main_model_path = os.path.join(base_dir, str(model_num), f'main_model.h5')
    main_model = tf.keras.models.load_model(main_model_path)

    data_filepath = r"C:\Users\proxi\Documents\coding\TOF_data\TOF_data\combined_data.h5"
    with h5py.File(data_filepath, 'r') as hf:
        test_data_ds = hf['combined_data'][:sample_size]
        mask = test_data_ds[:, -1].astype(bool)
        test_data_ds = test_data_ds[mask]

    #test_data_path = os.path.join(base_dir, str(model_num), 'test_data.h5')
    #test_data = load_test_data(test_data_path)

    # Load scalers
    scalers_path = os.path.join(base_dir, str(model_num), 'scalers.pkl')
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    # Prepare test data
    x_test_raw = test_data_ds[:, [2, 3, 4, 5]]  # Input features used during training
    y_test_raw = test_data_ds[:, 0]  # Energy (assuming column 0)

    # Apply log transformations
    epsilon = 1e-10
    y_test_log = np.log2(y_test_raw)
    x_test_log = x_test_raw.copy()
    x_test_log[:, -1] = np.log2(x_test_log[:, -1])

    # Process input data as during training
    generator = DataGeneratorTofToKE([], [], '')  # Dummy initialization
    x_test_processed = generator.process_input(x_test_raw)

    # Combine output and input data
    full_data = np.hstack([y_test_log.reshape(-1, 1), x_test_processed])

    # Scale the data
    scaled_full_data = scalers.transform(full_data)
    scaled_y_test = scaled_full_data[:, 0]
    scaled_x_test = scaled_full_data[:, 1:]

    # Make predictions
    y_pred_scaled = main_model.predict(scaled_x_test).flatten()

    # Inverse scaling
    y_pred_full_scaled = np.hstack([y_pred_scaled.reshape(-1, 1), scaled_x_test])
    y_pred_full = scalers.inverse_transform(y_pred_full_scaled)
    y_pred_log = y_pred_full[:, 0]

    # Inverse log transformation
    y_pred = 2 ** y_pred_log
    y_test = y_test_raw

    # Create DataFrame
    df_tof = pd.DataFrame({
        'y_tof_pred': y_pred.flatten(),
        'y_tof_test': y_test.flatten(),
        'tof_residuals': y_test.flatten() - y_pred.flatten(),
        'retardation': test_data_ds[:, 2].flatten()  # Adjust index if necessary
    })

    # Prepare to save plots to PDF if requested
    if pdf_filename:
        pdf_path = os.path.join(base_dir, str(model_num), pdf_filename)
        pdf_writer = PdfPages(pdf_path)

    # Scatter plot with regression line for time of flight
    fig, ax = plt.subplots(figsize=(8, 6))

    mae = mean_absolute_error(df_tof['y_tof_test'], df_tof['y_tof_pred'])
    rmse = np.sqrt(mean_squared_error(df_tof['y_tof_test'], df_tof['y_tof_pred']))
    r_squared = r2_score(df_tof['y_tof_test'], df_tof['y_tof_pred'])

    sns.scatterplot(x='y_tof_pred', y='y_tof_test', data=df_tof, hue='retardation', palette='viridis', alpha=0.7,
                    edgecolor='k', linewidth=0.5, s=80, ax=ax)
    sns.regplot(x='y_tof_pred', y='y_tof_test', data=df_tof, ci=95, scatter_kws={'alpha': 0},
                line_kws={"color": "red"},
                ax=ax)
    ax.set_xlabel('Predicted Values', fontsize=16)
    ax.set_ylabel('True Values', fontsize=16)
    ax.set_title('Time of Flight Performance')

    # Inset for metrics
    axins = inset_axes(ax, width="40%", height="30%", loc='upper right')
    axins.text(0.05, 0.7, f'MAE = {mae:.3f}', fontsize=14)
    axins.text(0.05, 0.5, f'RMSE = {rmse:.3f}', fontsize=14)
    axins.text(0.05, 0.3, f'$R^2 = {r_squared:.3f}$', fontsize=14)
    axins.axis('off')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Residual plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(x='y_tof_pred', y='tof_residuals', data=df_tof, hue='retardation', palette='viridis', alpha=0.7,
                    edgecolor='k', linewidth=0.5, s=80, ax=ax)
    ax.set_xlabel('Predicted Values', fontsize=16)
    ax.set_ylabel('Residuals', fontsize=16)
    ax.set_title('Time of Flight Residuals')

    plt.tight_layout()
    plt.show()
    if pdf_filename:
        pdf_writer.savefig(fig)
    plt.close(fig)

    # Histogram of residuals
    fig, ax = plt.subplots(figsize=(8, 6))

    col_res = 'tof_residuals'
    residual_kurtosis, residual_skewness = kurtosis(df_tof[col_res]), skew(df_tof[col_res])
    sns.histplot(df_tof[col_res], bins=30, kde=True, color='skyblue', edgecolor='k', linewidth=0.5, ax=ax)
    ax.set_xlabel('Residuals', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_title('Time of Flight Residuals Distribution')

    # Fit a normal distribution to residuals
    mu, std = norm.fit(df_tof[col_res])
    x = np.linspace(min(df_tof[col_res]), max(df_tof[col_res]), 100)
    pdf = norm.pdf(x, mu, std)

    # Scale the PDF to match the scale of the histogram
    bin_width = (max(df_tof[col_res]) - min(df_tof[col_res])) / 30
    scaled_pdf = pdf * len(df_tof[col_res]) * bin_width
    ax.plot(x, scaled_pdf, 'r-', lw=2)
    ax.legend(['Fitted Normal Distribution'])

    # Calculate FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * std

    # Inset for statistics
    axins_hist = inset_axes(ax, width="40%", height="40%", loc='upper right')
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
    plot_TOF_to_energy_results(args.base_dir, args.model_num, 1, pdf_filename=args.pdf_filename,
                 sample_size=args.sample_size)

if __name__ == '__main__':
    main()