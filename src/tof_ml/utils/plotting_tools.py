from pyqtgraph.Qt import QtGui, QtWidgets
from pyimagetool import ImageTool
import numpy as np
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import ks_2samp
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from pandas.plotting import parallel_coordinates
import xarray as xr
sys.path.insert(0, os.path.abspath('../..'))


def get_cmap(n, name='seismic'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


# src/utils/evaluation.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm, kurtosis, skew


# src/utils/evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle
from scipy.stats import norm, kurtosis, skew
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def evaluate_and_plot_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y=None,
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
    y_pred_scaled = model.predict(X_test).flatten()
    if scaler_y:
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_pred = y_pred_scaled
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




class PlotWindow(QtWidgets.QWidget):
    def __init__(self, plot_func, *args, **kwargs):
        super().__init__()
        self.plot_func = plot_func
        self.args = args
        self.kwargs = kwargs
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        plot_widget = self.plot_func(*self.args, **self.kwargs, layout=ImageTool.LayoutComplete)
        layout.addWidget(plot_widget)
        self.setLayout(layout)
        self.setWindowTitle('Plot')
        self.show()


def plot_imagetool(*args):
    app = QtWidgets.QApplication.instance()
    if app is None:
        # if it does not exist then a QApplication is created
        app = QtWidgets.QApplication([])
    for arg in args:
        window = PlotWindow(ImageTool, arg)
        app.exec_()
        window.close()

def random_sample_data(data, sample_size):
    indices = np.random.choice(len(data), min(sample_size, len(data)), replace=False)
    return np.array(data)[indices]

def plot_relation(ax, ds, x_param, y_param, x_label, y_label, title=None,
                  plot_log=False, retardation=None, kinetic_energy=None, mid1_ratio=None, mid2_ratio=None,
                  collection_efficiency=None, ks_score=None, verbose=False, sample_size=1000):
    def in_range(value, value_range):
        return value_range is None or (value_range[0] <= value <= value_range[1])

    data_to_plot = [
        item for item in ds
        if in_range(item['retardation'], retardation)
           and in_range(item['kinetic_energy'], kinetic_energy)
           and in_range(item['mid1_ratio'], mid1_ratio)
           and in_range(item['mid2_ratio'], mid2_ratio)
           and in_range(item['collection_efficiency'], collection_efficiency)
           and in_range(item['ks_score'], ks_score)
    ]
    data_to_plot = random_sample_data(data_to_plot, sample_size)

    grouped_data = {}
    for item in data_to_plot:
        key = (item['retardation'], item['mid1_ratio'], item['mid2_ratio'])
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(item)

    # Sort grouped_data by retardation in descending order
    sorted_grouped_data = sorted(grouped_data.items(), key=lambda x: x[0][0], reverse=True)

    colors = get_cmap(len(sorted_grouped_data))

    for idx, (key, group) in enumerate(sorted_grouped_data):
        color = colors(idx)
        for item in group:
            if int(item['retardation']) in [10, 7, 3, 1, -1, -3, -7, -10]:
                if not plot_log:
                    ax.scatter(item[x_param], item[y_param], alpha=0.6, color=color,
                               label=f"R={item['retardation']}, M1={item['mid1_ratio']}, M2={item['mid2_ratio']}")
                else:
                    ax.scatter(np.log2(item[x_param]), np.log2(item[y_param]), alpha=0.5, color=color,
                               label=f"R={item['retardation']}")

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {label: handle for handle, label in zip(handles, labels)}
    ax.legend(unique_labels.values(), unique_labels.keys())

    if verbose:
        tot = sum([len(d[y_param]) for d in data_to_plot])
        print(tot)

    if title:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)


def plot_parallel_coordinates(cuts, titles):
    """
    Plot parallel coordinates plots of 3D cuts from a 4D xarray with custom axes and colorbar.

    Parameters:
    cuts (list of xarray.DataArray): List of 3D cuts to plot.
    titles (list of str): List of titles for each plot.

    """
    num_plots = len(cuts)

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4), sharey=True)

    if num_plots == 1:
        axes = [axes]

    for ax, cut, title in zip(axes, cuts, titles):
        df = cut.to_dataframe(name='collection_efficiency').reset_index()
        kinetic_energy = df['kinetic_energy'].unique()

        # Create a unique identifier for each combination of mid1_ratio and mid2_ratio
        df['combo'] = list(zip(df['mid1_ratio'], df['mid2_ratio']))
        unique_combos = df['combo'].unique()

        # Create a colormap
        norm = mcolors.Normalize(vmin=0, vmax=len(unique_combos) - 1)
        cmap = cm.get_cmap('viridis', len(unique_combos))

        # Create a mapping from combo to color index
        combo_to_index = {combo: idx for idx, combo in enumerate(unique_combos)}

        for combo, group in df.groupby('combo'):
            color_idx = combo_to_index[combo]
            color = cmap(color_idx)
            ax.plot(kinetic_energy, group['collection_efficiency'].values,
                    label=f'M1: {combo[0]}, M2: {combo[1]}',
                    color=color)
            # Draw vertical lines
            for k in kinetic_energy:
                ax.axvline(x=k, color='gray', linestyle='--', linewidth=0.5)

        ax.set_title(title)
        ax.set_xlabel('Kinetic Energy')

    axes[0].set_ylabel('Collection Efficiency')

    # Create a colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Combination Index')

    # Create a table next to the plot
    table_data = [(combo_to_index[combo], combo[0], combo[1]) for combo in unique_combos]
    table_df = pd.DataFrame(table_data, columns=['Index', 'mid1_ratio', 'mid2_ratio'])

    fig_table, ax_table = plt.subplots(figsize=(3, 4))
    ax_table.axis('tight')
    ax_table.axis('off')
    table = ax_table.table(cellText=table_df.values, colLabels=table_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    plt.show()

def plot_ks_score(data_masked, retardations, mid1, mid2, R=13.74, bootstrap=None, directory=None,
                  filename=None, kinetic_energies=None):
    """
    Plot particle distributions for each retardation, colored by kinetic energy.

    Parameters:
    data_masked (list of dict): Masked data loaded by DS_positive.
    mid1 (float): Mid1 value to display in the title.
    mid2 (float): Mid2 value to display in the title.
    R (float): Radius for the uniform distribution.
    bootstrap (int): Number of bootstrap samples to use for calculating KS scores.
    directory (str): Directory to save the PDF file.
    filename (str): Filename for the PDF file.
    kinetic_energies (list or None): List of specific kinetic energies to include in the plot.

    Returns:
    dict: A dictionary of average KS scores for each retardation.
    """
    ks_scores = {}
    save_to_pdf = directory is not None and filename is not None

    if save_to_pdf:
        pdf_path = os.path.join(directory, filename)
        pdf_pages = PdfPages(pdf_path)

    for retardation in retardations:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Filter data for the current retardation
        filtered_data = [entry for entry in data_masked if entry['retardation'] == retardation and
                         entry['mid1_ratio'] == mid1 and entry['mid2_ratio'] == mid2]

        # Filter data by specified kinetic energies if provided
        if kinetic_energies is not None:
            filtered_data = [entry for entry in filtered_data if entry['kinetic_energy'] in kinetic_energies]
            for ke in kinetic_energies:
                if not any(entry['kinetic_energy'] == ke for entry in filtered_data):
                    print(f"ke {ke} is not in the list of kinetic energies")

        if not filtered_data:
            continue

        # Get unique kinetic energies and create a color map
        kinetic_energies_in_data = sorted(list(set(entry['kinetic_energy'] for entry in filtered_data)))
        norm = Normalize(vmin=min(kinetic_energies_in_data), vmax=max(kinetic_energies_in_data))
        cmap = plt.cm.viridis

        ks_score_dict = {}

        for entry in filtered_data:
            y_pos = entry['y_tof']
            num_points = len(y_pos)

            if num_points == 0:
                continue

            theta_uniform = np.random.uniform(0, 2 * np.pi, num_points)
            radius_uniform = np.random.uniform(0, R ** 2, num_points) ** 0.5
            x_uniform = radius_uniform * np.cos(theta_uniform)
            y_uniform = radius_uniform * np.sin(theta_uniform)

            theta_final = np.random.uniform(0, 2 * np.pi, num_points)
            radius_final = np.abs(y_pos)
            x_final = radius_final * np.cos(theta_final)
            y_final = radius_final * np.sin(theta_final)

            uniform_dist = np.hstack((x_uniform, y_uniform))
            final_dist = np.hstack((x_final, y_final))

            if bootstrap:
                bootstrap_scores = []
                for _ in range(bootstrap):
                    sample_indices = np.random.choice(num_points, num_points, replace=True)
                    uniform_sample = uniform_dist[sample_indices]
                    final_sample = final_dist[sample_indices]
                    ks_score, _ = ks_2samp(uniform_sample, final_sample)
                    bootstrap_scores.append(ks_score)
                ks_score = np.mean(bootstrap_scores)
            else:
                ks_score, _ = ks_2samp(uniform_dist, final_dist)

            kinetic_energy = entry['kinetic_energy']

            if retardation not in ks_scores:
                ks_scores[retardation] = []
            ks_scores[retardation].append(ks_score)

            if kinetic_energy not in ks_score_dict:
                ks_score_dict[kinetic_energy] = []
            ks_score_dict[kinetic_energy].append(ks_score)

            color = cmap(norm(kinetic_energy))
            ax[0].scatter(x_uniform, y_uniform, alpha=0.5, color=color,
                          label=f'KE: {kinetic_energy:.2f}, KS: {ks_score:.2f}')
            ax[1].scatter(x_final, y_final, alpha=0.5, color=color,
                          label=f'KE: {kinetic_energy:.2f}, KS: {ks_score:.2f}')

        title_sign = '+' if retardation >= 0 else '-'
        fig.suptitle(f'Detector distributions for {title_sign}{abs(retardation):.2f} eV and {mid1} at Blade 22 and {mid2} at Blade 25')

        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_aspect('equal', 'box')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        ax[1].set_aspect('equal', 'box')
        # Add legend only if there are valid labels
        handles, labels = ax[1].get_legend_handles_labels()
        if handles:
            legend_order = sorted(zip(handles, labels), key=lambda x: float(x[1].split()[1][:-1]))
            ordered_handles, ordered_labels = zip(*legend_order)
            ax[1].legend(ordered_handles, ordered_labels, loc='upper right')

        # Adjust layout to make space for the colorbar
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        # Create the colorbar outside the plot
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.6])
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, label='Kinetic Energy')

        if save_to_pdf:
            pdf_pages.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

    if save_to_pdf:
        pdf_pages.close()

    # Calculate average KS scores for each retardation
    avg_ks_scores = {ret: np.nanmean(scores) for ret, scores in ks_scores.items()}

    return avg_ks_scores


def plot_collection_efficiency(xar, retardations, mid1_ratios, mid2_ratios, directory=None, filename=None):
    colors = plt.cm.viridis(np.linspace(0, 1, len(mid1_ratios)))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    if directory and filename:
        output_path = os.path.join(directory, f'{filename}.pdf')
        pdf_pages = PdfPages(output_path)

    for retardation in retardations:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if retardation > 0:
            ke = 0.1
            title_retardation = f'+{retardation} eV Retardation'
        else:
            ke = np.abs(retardation) + 2
            title_retardation = f'-{np.abs(retardation)} eV Retardation'

        # First plot: mid1 and mid2 as x and y axes for a specific kinetic energy above the retardation
        cut = xar.sel({'kinetic_energy': ke, 'retardation': retardation})
        cut.plot(x='mid1_ratio', y='mid2_ratio', ax=ax1)

        ax1.set_title(f'Collection Efficiency for {title_retardation} and {ke} eV KE')
        ax1.set_xlabel('mid1_ratio')
        ax1.set_ylabel('mid2_ratio')
        ax1.grid(True)

        # Second plot: Kinetic energy on x axis with mid1 as the same color but different linestyle for each mid2
        for i, mid1_ratio in enumerate(mid1_ratios):
            for j, mid2_ratio in enumerate(mid2_ratios):
                cut = xar.sel({'retardation': retardation, 'mid1_ratio': mid1_ratio, 'mid2_ratio': mid2_ratio})
                cut.plot(ax=ax2, color=colors[i], linestyle=linestyles[j % len(linestyles)])

        ax2.set_title(f'Collection Efficiency vs. Kinetic Energy for {title_retardation}')
        ax2.set_xlabel('Kinetic Energy (eV)')
        ax2.set_ylabel('Collection Efficiency')
        ax2.grid(True)

        # Create custom legend entries
        handles1 = [plt.Line2D([0], [0], color=colors[i], lw=4, label=f'mid1: {mid1_ratio}') for i, mid1_ratio in enumerate(mid1_ratios)]
        handles2 = [plt.Line2D([0], [0], color='black', linestyle=linestyles[j % len(linestyles)], label=f'mid2: {mid2_ratio}') for j, mid2_ratio in enumerate(mid2_ratios)]

        # Adding legends
        ax2.legend(handles=handles1 + handles2, loc='best')

        if directory and filename:
            pdf_pages.savefig(fig)
            plt.close()
        else:
            plt.show()

    if directory and filename:
        pdf_pages.close()


def plot_collection_efficiency_grid(xar, retardations, kinetic_energies, directory=None, filename=None):
    num_plots = len(retardations)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed

    if directory and filename:
        output_path = os.path.join(directory, f'{filename}.pdf')
        pdf_pages = PdfPages(output_path)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    axes = axes.flatten()

    for idx, (retardation, ke) in enumerate(zip(retardations, kinetic_energies)):
        ax = axes[idx]
        cut = xar.sel({'kinetic_energy': ke, 'retardation': retardation})
        im = cut.plot(x='mid1_ratio', y='mid2_ratio', ax=ax, vmin=0, vmax=1, cmap='bwr')

        if retardation > 0:
            title_retardation = f'R = +{retardation} V, KE = {ke} eV'
        else:
            title_retardation = f'R = {retardation} V, KE = {ke} eV'

        ax.set_title(title_retardation)
        ax.set_xlabel('Blade 22')
        ax.set_ylabel('Blade 25')
        ax.grid(True)

    # Hide any remaining empty subplots
    for idx in range(len(retardations), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    if directory and filename:
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close()
        pdf_pages.close()
    else:
        plt.show()


def plot_xar_instances(xar, retardations, mid1_ratios, mid2_ratios, value_def="Collection Efficiency", directory=None, filename=None):
    colors = plt.cm.viridis(np.linspace(0, 1, len(mid1_ratios)))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    if directory and filename:
        output_path = os.path.join(directory, f'{filename}.pdf')
        pdf_pages = PdfPages(output_path)

    for retardation in retardations:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if retardation > 0:
            ke = 10
            title_retardation = f'+{retardation} eV Retardation'
        else:
            ke = np.abs(retardation) + 10
            title_retardation = f'-{np.abs(retardation)} eV Retardation'

        # First plot: mid1 and mid2 as x and y axes for a specific kinetic energy above the retardation
        cut = xar.sel({'kinetic_energy': ke, 'retardation': retardation})
        cut.plot(x='mid1_ratio', y='mid2_ratio', ax=ax1, cmap='bwr')

        ax1.set_title(f'{value_def} for {title_retardation} and {ke} eV KE')
        ax1.set_xlabel('Blade 22')
        ax1.set_ylabel('Blade 25')
        ax1.grid(True)

        # Second plot: Kinetic energy on x axis with mid1 as the same color but different linestyle for each mid2
        for i, mid1_ratio in enumerate(mid1_ratios):
            for j, mid2_ratio in enumerate(mid2_ratios):
                cut = xar.sel({'retardation': retardation, 'mid1_ratio': mid1_ratio, 'mid2_ratio': mid2_ratio})
                cut.plot(ax=ax2, color=colors[i], linestyle=linestyles[j % len(linestyles)])

        ax2.set_title(f'{value_def} vs. Kinetic Energy for {title_retardation}')
        ax2.set_xlabel('Kinetic Energy (eV)')
        ax2.set_ylabel(f'{value_def}')
        ax2.grid(True)

        # Create custom legend entries
        handles1 = [plt.Line2D([0], [0], color=colors[i], lw=4, label=f'Blade 22: {mid1_ratio}')
                    for i, mid1_ratio in enumerate(mid1_ratios)]
        handles2 = [plt.Line2D([0], [0], color='black', linestyle=linestyles[j % len(linestyles)],
                               label=f'Blade 25: {mid2_ratio}') for j, mid2_ratio in enumerate(mid2_ratios)]

        # Adding legends
        ax2.legend(handles=handles1 + handles2, loc='best')

        if directory and filename:
            pdf_pages.savefig(fig)
            plt.close()
        else:
            plt.show()

    if directory and filename:
        pdf_pages.close()

def plot_xar_single_axis(xar, retardations, ratios, value_def="Collection Efficiency", directory=None, filename=None):
    colors = plt.cm.viridis(np.linspace(0, 1, len(retardations)))
    linestyles = ['-', '--', ':']
    lineweights = np.linspace(2, 3, len(retardations))  # Adjust line weights from 1 to 3

    if directory and filename:
        output_path = os.path.join(directory, f'{filename}.pdf')
        pdf_pages = PdfPages(output_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, retardation in enumerate(retardations):
        for j, (mid1_ratio, mid2_ratio) in enumerate(ratios):
            cut = xar.sel({'retardation': retardation, 'mid1_ratio': mid1_ratio, 'mid2_ratio': mid2_ratio})
            cut.plot(ax=ax, color=colors[i], linestyle=linestyles[j % len(linestyles)],
                     linewidth=lineweights[i], label=f'Ret: {retardation}\nM1: {mid1_ratio}, M2: {mid2_ratio}')

    ax.set_title(f'{value_def} vs. Kinetic Energy')
    ax.set_xlabel('Kinetic Energy (eV)')
    ax.set_ylabel(f'{value_def}')
    ax.set_ylim([0, 1])
    ax.grid(True)

    # Create custom legend entries
    handles = [plt.Line2D([0], [0], color=colors[i], linestyle=linestyles[j % len(linestyles)], linewidth=lineweights[i],
                          label=f'Ret: {retardation}\nM1: {mid1_ratio}, M2: {mid2_ratio}')
               for i, retardation in enumerate(retardations)
               for j, (mid1_ratio, mid2_ratio) in enumerate(ratios)]

    # Adding legends outside the plot
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')

    plt.tight_layout()

    if directory and filename:
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close()
        pdf_pages.close()
    else:
        plt.show()


def plot_heatmap(collection_efficiency):
    data = []
    for retardation, energies in collection_efficiency.items():
        for kinetic_energy, efficiency in energies.items():
            data.append([retardation, kinetic_energy, efficiency])

    df = pd.DataFrame(data, columns=['Retardation', 'Kinetic Energy', 'Collection Efficiency'])
    heatmap_data = df.pivot('Kinetic Energy', 'Retardation', 'Collection Efficiency')

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', cbar_kws={'label': 'Collection Efficiency'})
    plt.title('Heatmap of Collection Efficiency')
    plt.xlabel('Retardation')
    plt.ylabel('Kinetic Energy')
    plt.show()


def plot_histogram(ax, ds, parameter, x_label, title=None, nbins=50, selected_pairs=None):
    if selected_pairs:
        cmap = get_cmap(len(selected_pairs))
    else:
        cmap = get_cmap(len(ds))
    color_index = 0
    print(selected_pairs)

    for i in range(len(ds)):
        retardation = ds[i]['retardation']
        kinetic_energy = ds[i]['kinetic_energy']
        mid1_ratio = ds[i]['mid1_ratio']
        mid2_ratio = ds[i]['mid2_ratio']
        if selected_pairs is None or (retardation, kinetic_energy, mid1_ratio, mid2_ratio) in selected_pairs:
            collection_efficiency = ds[i]['collection_efficiency']
            ks_score = ds[i]['ks_score']
            ax.hist(ds[i][parameter], bins=nbins, color=cmap(color_index),
                    edgecolor='black', alpha=0.5,
                    label=f"R={retardation}, KE={kinetic_energy}, CE={collection_efficiency}, KS score={ks_score:.2f}, "
                          f"M1={mid1_ratio}, M2={mid2_ratio}")
            ax.legend(fontsize=8)
            color_index += 1

    if title:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)


def plot_energy_resolution(gradients, retardation, mid1_range, mid2_range):
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    colors = plt.cm.viridis(np.linspace(0, 1, len(gradients)))
    color_index = 0

    for (r, mid1, mid2), (avg_tof_values, kinetic_energies, gradient, error) in gradients.items():
        if r == retardation and mid1_range[0] <= mid1 <= mid1_range[1] and mid2_range[0] <= mid2 <= mid2_range[1]:
            kinetic_energies = sorted(kinetic_energies)
            label = f'Mid1: {mid1}, Mid2: {mid2}'
            axes[0].scatter(kinetic_energies, avg_tof_values, label=label)
            axes[1].scatter(kinetic_energies, gradient, label=label)

    axes[0].set_title(f'TOF values vs Initial KE for Retardation = {retardation}')
    axes[0].set_xlabel('Initial KE')
    axes[0].set_ylabel('TOF values')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title(f'Gradient of TOF values vs Initial KE for Retardation = {retardation}')
    axes[1].set_xlabel('Initial KE')
    axes[1].set_ylabel('Gradient of TOF values')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

