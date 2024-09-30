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


def get_cmap(n, name='seismic'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


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
                  plot_log=False, retardation=None, pass_energy=None, mid1_ratio=None, mid2_ratio=None,
                  collection_efficiency=None, ks_score=None, verbose=False, sample_size=1000):
    def in_range(value, value_range):
        return value_range is None or (value_range[0] <= value <= value_range[1])

    data_to_plot = [
        item for item in ds
        if in_range(item['retardation'], retardation)
           and in_range(item['pass_energy'], pass_energy)
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
        pass_energy = df['pass_energy'].unique()

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
            ax.plot(pass_energy, group['collection_efficiency'].values,
                    label=f'M1: {combo[0]}, M2: {combo[1]}',
                    color=color)
            # Draw vertical lines
            for k in pass_energy:
                ax.axvline(x=k, color='gray', linestyle='--', linewidth=0.5)

        ax.set_title(title)
        ax.set_xlabel('Pass Energy')

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
                  filename=None, pass_energies=None, pdf_pages=None):
    """
    Plot particle distributions for each retardation, colored by Pass Energy.

    Parameters:
    data_masked (list of dict): Masked data loaded by DataProcessor.
    mid1 (float): Mid1 value to display in the title.
    mid2 (float): Mid2 value to display in the title.
    R (float): Radius for the uniform distribution.
    bootstrap (int): Number of bootstrap samples to use for calculating KS scores.
    directory (str): Directory to save the PDF file.
    filename (str): Filename for the PDF file.
    pass_energies (list or None): List of specific pass energies to include in the plot.
    pdf_pages (PdfPages or None): PdfPages object to save plots into. If provided, plots will be saved into this PDF.

    Returns:
    dict: A dictionary of average KS scores for each retardation.
    """
    ks_scores = {}
    save_to_pdf = (directory is not None and filename is not None) or pdf_pages is not None

    # Only create a new PdfPages object if one is not provided
    if save_to_pdf and pdf_pages is None:
        pdf_path = os.path.join(directory, filename)
        pdf_pages = PdfPages(pdf_path)

    for retardation in retardations:
        fig, ax = plt.subplots(1, 2, figsize=(12, 7))

        # Filter data for the current retardation
        filtered_data = [
            entry for entry in data_masked
            if np.isclose(entry['retardation'], retardation) and
               np.isclose(entry['mid1_ratio'], mid1) and
               np.isclose(entry['mid2_ratio'], mid2)
        ]
        print(f"Number of entries for retardation {retardation}, mid1 {mid1}, mid2 {mid2}: {len(filtered_data)}")

        # Filter data by specified pass energies if provided
        if pass_energies is not None:
            pass_energies = [np.round(float(pe), decimals=2) for pe in pass_energies]
            pass_energies_array = np.array(pass_energies)  # Convert to Numpy array
            filtered_data = [
                entry for entry in filtered_data
                if np.any(np.isclose(entry['pass_energy'], pass_energies_array))
            ]
            for pe in pass_energies:
                if not any(np.isclose(entry['pass_energy'], pe) for entry in filtered_data):
                    print(f"Pass energy {pe} is not in the list of pass energies for retardation {retardation}")

        if not filtered_data:
            plt.close(fig)
            continue

        # Get unique pass energies and create a color map
        pass_energies_in_data = sorted(list(set(entry['pass_energy'] for entry in filtered_data)))
        norm = Normalize(vmin=min(pass_energies_in_data), vmax=max(pass_energies_in_data))
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
                    ks_score_sample, _ = ks_2samp(uniform_sample, final_sample)
                    bootstrap_scores.append(ks_score_sample)
                ks_score = np.mean(bootstrap_scores)
            else:
                ks_score, _ = ks_2samp(uniform_dist, final_dist)

            pass_energy = entry['pass_energy']

            if retardation not in ks_scores:
                ks_scores[retardation] = []
            ks_scores[retardation].append(ks_score)

            if pass_energy not in ks_score_dict:
                ks_score_dict[pass_energy] = []
            ks_score_dict[pass_energy].append(ks_score)

            color = cmap(norm(pass_energy))
            # Include CE, CE rank, and KS score rank in the label
            ce = entry['collection_efficiency']
            ce_rank = entry.get('ce_rank', None)
            ks_score_rank = entry.get('ks_score_rank', None)
            # Build the label string conditionally
            label_parts = [
                f'PE: {pass_energy:.2f}',
                f'KS: {ks_score:.2f}',
                f'CE: {ce:.2f}'
            ]
            if ce_rank is not None:
                label_parts.append(f'\nCE Rank: {ce_rank:.2f}')
            if ks_score_rank is not None:
                label_parts.append(f'KS Rank: {ks_score_rank:.2f}')
            label = ', '.join(label_parts)
            ax[0].scatter(x_uniform, y_uniform, alpha=0.5, color=color,
                          label=label)
            ax[1].scatter(x_final, y_final, alpha=0.5, color=color,
                          label=label)

        title_sign = '+' if retardation >= 0 else '-'
        fig.suptitle(f'Detector distributions for {title_sign}{abs(retardation):.2f} eV\nmid1: {mid1}, mid2: {mid2}')

        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_aspect('equal', 'box')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        ax[1].set_aspect('equal', 'box')

        # Adjust layout to make space for the legend at the bottom
        plt.tight_layout(rect=[0.05, 0.15, 0.85, 0.95])  # Adjust the rect to leave space at the bottom

        # Move the legend to the bottom center
        handles, labels = ax[1].get_legend_handles_labels()
        if handles:
            # Sort the legend entries based on pass energy
            def extract_pe(label):
                try:
                    return float(label.split(',')[0].split(':')[1].strip())
                except ValueError:
                    return 0.0  # Default if parsing fails

            legend_order = sorted(zip(handles, labels), key=lambda x: extract_pe(x[1]))
            ordered_handles, ordered_labels = zip(*legend_order)
            n_entries = len(ordered_handles)
            n_cols = min(n_entries, 3)  # Adjust the number of columns, up to 3
            legend = fig.legend(
                ordered_handles, ordered_labels,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.02),
                ncol=n_cols,
                frameon=False
            )
            # Optionally adjust the font size
            for text in legend.get_texts():
                text.set_fontsize('small')  # Adjust as needed

        # Create the colorbar outside the plot
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # Adjust the position to avoid overlapping with legend
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, label='Pass Energy')

        if save_to_pdf:
            pdf_pages.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

    # Close the PDF only if we opened it in this function
    if save_to_pdf and pdf_pages is not None and directory is not None and filename is not None:
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
            pe = 0.1
            title_retardation = f'+{retardation} eV Retardation'
        else:
            pe = np.abs(retardation) + 2
            title_retardation = f'-{np.abs(retardation)} eV Retardation'

        # First plot: mid1 and mid2 as x and y axes for a specific Pass Energy above the retardation
        cut = xar.sel({'pass_energy': pe, 'retardation': retardation})
        cut.plot(x='mid1_ratio', y='mid2_ratio', ax=ax1)

        ax1.set_title(f'Collection Efficiency for {title_retardation} and {pe} eV pe')
        ax1.set_xlabel('mid1_ratio')
        ax1.set_ylabel('mid2_ratio')
        ax1.grid(True)

        # Second plot: Pass Energy on x axis with mid1 as the same color but different linestyle for each mid2
        for i, mid1_ratio in enumerate(mid1_ratios):
            for j, mid2_ratio in enumerate(mid2_ratios):
                cut = xar.sel({'retardation': retardation, 'mid1_ratio': mid1_ratio, 'mid2_ratio': mid2_ratio})
                cut.plot(ax=ax2, color=colors[i], linestyle=linestyles[j % len(linestyles)])

        ax2.set_title(f'Collection Efficiency vs. Pass Energy for {title_retardation}')
        ax2.set_xlabel('Pass Energy (eV)')
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


def plot_collection_efficiency_grid(xar, retardations, pass_energies, directory=None, filename=None):
    num_plots = len(retardations)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed

    if directory and filename:
        output_path = os.path.join(directory, f'{filename}.pdf')
        pdf_pages = PdfPages(output_path)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    axes = axes.flatten()

    for idx, (retardation, pe) in enumerate(zip(retardations, pass_energies)):
        ax = axes[idx]
        cut = xar.sel({'pass_energy': pe, 'retardation': retardation})
        im = cut.plot(x='mid1_ratio', y='mid2_ratio', ax=ax, vmin=0, vmax=1, cmap='bwr')

        if retardation > 0:
            title_retardation = f'R = +{retardation} V, pe = {pe} eV'
        else:
            title_retardation = f'R = {retardation} V, pe = {pe} eV'

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
            pe = 10
            title_retardation = f'+{retardation} eV Retardation'
        else:
            pe = np.abs(retardation) + 10
            title_retardation = f'-{np.abs(retardation)} eV Retardation'

        # First plot: mid1 and mid2 as x and y axes for a specific Pass Energy above the retardation
        cut = xar.sel({'pass_energy': pe, 'retardation': retardation})
        cut.plot(x='mid1_ratio', y='mid2_ratio', ax=ax1, cmap='bwr')

        ax1.set_title(f'{value_def} for {title_retardation} and {pe} eV pe')
        ax1.set_xlabel('Blade 22')
        ax1.set_ylabel('Blade 25')
        ax1.grid(True)

        # Second plot: Pass Energy on x axis with mid1 as the same color but different linestyle for each mid2
        for i, mid1_ratio in enumerate(mid1_ratios):
            for j, mid2_ratio in enumerate(mid2_ratios):
                cut = xar.sel({'retardation': retardation, 'mid1_ratio': mid1_ratio, 'mid2_ratio': mid2_ratio})
                cut.plot(ax=ax2, color=colors[i], linestyle=linestyles[j % len(linestyles)])

        ax2.set_title(f'{value_def} vs. Pass Energy for {title_retardation}')
        ax2.set_xlabel('Pass Energy (eV)')
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

    ax.set_title(f'{value_def} vs. Pass Energy')
    ax.set_xlabel('Pass Energy (eV)')
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
        for pass_energy, efficiency in energies.items():
            data.append([retardation, pass_energy, efficiency])

    df = pd.DataFrame(data, columns=['Retardation', 'Pass Energy', 'Collection Efficiency'])
    heatmap_data = df.pivot('Pass Energy', 'Retardation', 'Collection Efficiency')

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', cbar_kws={'label': 'Collection Efficiency'})
    plt.title('Heatmap of Collection Efficiency')
    plt.xlabel('Retardation')
    plt.ylabel('Pass Energy')
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
        pass_energy = ds[i]['pass_energy']
        mid1_ratio = ds[i]['mid1_ratio']
        mid2_ratio = ds[i]['mid2_ratio']
        if selected_pairs is None or (retardation, pass_energy, mid1_ratio, mid2_ratio) in selected_pairs:
            collection_efficiency = ds[i]['collection_efficiency']
            ks_score = ds[i]['ks_score']
            ax.hist(ds[i][parameter], bins=nbins, color=cmap(color_index),
                    edgecolor='black', alpha=0.5,
                    label=f"R={retardation}, KE={pass_energy}, CE={collection_efficiency}, KS score={ks_score:.2f}, "
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

    for (r, mid1, mid2), (avg_tof_values, pass_energies, gradient, error) in gradients.items():
        if r == retardation and mid1_range[0] <= mid1 <= mid1_range[1] and mid2_range[0] <= mid2 <= mid2_range[1]:
            pass_energies = sorted(pass_energies)
            label = f'Mid1: {mid1}, Mid2: {mid2}'
            axes[0].scatter(pass_energies, avg_tof_values, label=label)
            axes[1].scatter(pass_energies, gradient, label=label)

    axes[0].set_title(f'TOF values vs Initial KE for Retardation = {retardation}')
    axes[0].set_xlabel('Initial pe')
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

