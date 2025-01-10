import tensorflow as tf
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kurtosis, skew, norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages



def evaluate(model, x_test, y_test, scaler_y, plot=True):
    """
    Evaluates the trained model on the test set and optionally plots the results.

    Parameters:
    - model: Trained machine learning model.
    - x_test (np.ndarray): Scaled test input features.
    - y_test (np.ndarray): Scaled test outputs.
    - scaler_y (StandardScaler): Fitted scaler for the output variable.
    - plot (bool): Whether to generate evaluation plots.

    Returns:
    - loss_test (float): Evaluation metric (e.g., RMSE).
    """
    # Evaluate the model using the scaled test data
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    
    if plot:
        # Predict on the test data
        y_pred_scaled = model.predict(x_test).flatten()
        
        # Inverse transform the scaled predictions and true values
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Assume 'retardation' is the first feature in x_test
        retardation_values = x_test[:, 0]
        df = pd.DataFrame({
            'y_pred': y_pred.tolist(),
            'y_test': y_test.tolist(),
            'residuals': residuals.tolist(),
            'retardation': retardation_values.tolist()
        })

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r_squared = r2_score(y_test, y_pred)

        # Scatter plot with regression line
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_retardation = np.unique(retardation_values)
        sns.scatterplot(x='y_pred', y='y_test', data=df,
                        hue='retardation', palette='viridis', hue_order=unique_retardation,
                        alpha=0.7, edgecolor='k', linewidth=0.5, s=80)
        sns.regplot(x='y_pred', y='y_test', data=df, ci=95,
                    scatter_kws={'alpha': 0},  # Hide the original scatter to show the colored one
                    line_kws={"color": "red"})
        plt.xlabel('True Values', fontsize=16)
        plt.ylabel('Predicted Values', fontsize=16)
        # plt.title('True vs. Predicted Values with Retardation Color Coding', fontsize=14)

        # Inset for metrics
        axins = inset_axes(plt.gca(), width="40%", height="30%", loc='lower right')
        axins.text(0.05, 0.7, f'MAE = {mae:.3f}', fontsize=14)
        axins.text(0.05, 0.5, f'RMSE = {rmse:.3f}', fontsize=14)
        axins.text(0.05, 0.3, f'$R^2 = {r_squared:.3f}$', fontsize=14)
        axins.axis('off')
        plt.tight_layout()
        plt.show()
        # pdf_writer.add_page(fig_to_pdf_page(fig))
        # plt.close(fig)

        # Residual plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='y_pred', y='residuals', data=df,
                        hue='retardation', palette='viridis', hue_order=unique_retardation,
                        alpha=0.7, edgecolor='k', linewidth=0.5, s=80)
        plt.xlabel('Predicted Values', fontsize=16)
        plt.ylabel('Residuals', fontsize=16)
        # plt.title('Residual Plot with Retardation Color Coding', fontsize=16)
        # pdf_writer.add_page(fig_to_pdf_page(fig))
        # plt.close(fig)
        plt.show()

        # Histogram of residuals with fitted normal distribution
        residual_kurtosis, residual_skewness = kurtosis(residuals), skew(residuals)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, bins=30, kde=True, color='skyblue',
                     edgecolor='k', linewidth=0.5)
        plt.xlabel('Residuals', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        # plt.title('Histogram of Residuals', fontsize=16)

        # Fit a normal distribution to residuals
        mu, std = norm.fit(residuals)
        x = np.linspace(min(residuals), max(residuals), 100)
        pdf = norm.pdf(x, mu, std)

        # Scale the PDF to match the scale of the histogram
        bin_width = (max(residuals) - min(residuals)) / 30
        scaled_pdf = pdf * len(residuals) * bin_width
        plt.plot(x, scaled_pdf, 'r-', lw=2)
        plt.legend(['Fitted Normal Distribution'])

        # Calculate FWHM
        fwhm = 2 * np.sqrt(2 * np.log(2)) * std

        # Inset for kurtosis, skewness, center, and FWHM, with grid and axes removed
        axins_hist = inset_axes(plt.gca(), width="40%", height="40%", loc='upper left')
        axins_hist.text(0.3, 0.7, f'Kurtosis = {residual_kurtosis:.2f}', fontsize=12)
        axins_hist.text(0.3, 0.5, f'Skewness = {residual_skewness:.2f}', fontsize=12)
        axins_hist.text(0.3, 0.3, f'Center = {mu:.2f}', fontsize=12)
        axins_hist.text(0.3, 0.1, f'FWHM = {fwhm:.2f}', fontsize=12)
        axins_hist.axis('off')  # This turns off the axis labels and grid
        plt.show()
    return test_loss
