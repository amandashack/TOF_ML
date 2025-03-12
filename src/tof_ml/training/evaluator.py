# src/utils/evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm, kurtosis, skew
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def evaluate_and_plot_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "./test_plots",
    prefix: str = "test"
):
    """
    Evaluates the trained model on the test set, generates three plots:
      1. True vs. Predicted (inset: MAE, MSE, RMSE, R^2)
      2. Residuals vs. Predicted (inset: MAE, MSE, RMSE)
      3. Histogram of Residuals (inset: kurtosis, skewness, center, FWHM)

    Returns:
      (test_mse, plot_paths)
        test_mse: float
        plot_paths: dict of { "true_vs_pred": "...", "residuals": "...", "histogram_residuals": "..." }
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Predict
    y_pred = model.predict(X_test).flatten()
    y_true = y_test

    # 2. Residuals
    residuals = y_true - y_pred

    # 3. Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2_val = r2_score(y_true, y_pred)

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "residuals": residuals
    })

    # --------------------------------------------------------------------------
    # Plot 1: True vs. Predicted
    # --------------------------------------------------------------------------
    true_vs_pred_path = os.path.join(output_dir, f"{prefix}_true_vs_pred.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x="y_true", y="y_pred", data=df, alpha=0.7, edgecolor='k', linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal 1:1')
    ax.legend()
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Test: True vs. Predicted")

    # Inset for metrics
    axins = inset_axes(ax, width="35%", height="30%", loc='lower right')
    axins.axis('off')
    axins.text(0.1, 0.7, f"MAE  = {mae:.3f}")
    axins.text(0.1, 0.5, f"MSE  = {mse:.3f}")
    axins.text(0.1, 0.3, f"RMSE = {rmse:.3f}")
    axins.text(0.1, 0.1, f"R^2  = {r2_val:.3f}")

    plt.savefig(true_vs_pred_path, dpi=150)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Plot 2: Residuals vs. Predicted
    # --------------------------------------------------------------------------
    residuals_path = os.path.join(output_dir, f"{prefix}_residuals.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x="y_pred", y="residuals", data=df, alpha=0.7, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (True - Pred)")
    ax.set_title("Test: Residuals vs. Predicted")

    axins2 = inset_axes(ax, width="35%", height="30%", loc='lower right')
    axins2.axis('off')
    axins2.text(0.1, 0.7, f"MAE  = {mae:.3f}")
    axins2.text(0.1, 0.5, f"MSE  = {mse:.3f}")
    axins2.text(0.1, 0.3, f"RMSE = {rmse:.3f}")

    plt.savefig(residuals_path, dpi=150)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Plot 3: Histogram of Residuals
    # --------------------------------------------------------------------------
    hist_path = os.path.join(output_dir, f"{prefix}_hist_residuals.png")
    res_kurt = kurtosis(residuals)
    res_skew = skew(residuals)
    mu, std = norm.fit(residuals)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * std

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=False, color='skyblue', edgecolor='k', linewidth=0.5)
    ax.set_title("Histogram of Residuals (Test)")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")

    x_vals = np.linspace(residuals.min(), residuals.max(), 100)
    pdf_vals = norm.pdf(x_vals, mu, std)
    bin_width = (residuals.max() - residuals.min()) / 30
    scaled_pdf = pdf_vals * len(residuals) * bin_width
    ax.plot(x_vals, scaled_pdf, 'r-', lw=2, label="Fitted Normal PDF")
    ax.legend()

    axins3 = inset_axes(ax, width="40%", height="40%", loc='upper left')
    axins3.axis('off')
    axins3.text(0.1, 0.7,  f"Kurtosis = {res_kurt:.2f}")
    axins3.text(0.1, 0.5,  f"Skewness = {res_skew:.2f}")
    axins3.text(0.1, 0.3,  f"Center   = {mu:.2f}")
    axins3.text(0.1, 0.1,  f"FWHM     = {fwhm:.2f}")

    plt.savefig(hist_path, dpi=150)
    plt.close(fig)

    plot_paths = {
        "true_vs_pred": true_vs_pred_path,
        "residuals": residuals_path,
        "histogram_residuals": hist_path
    }

    return mse, plot_paths
