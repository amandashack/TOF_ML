from loaders.general_dataloader import DataProcessor, custom_data_loader
from utilities.plotting_tools import plot_ks_score
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

# Assuming random_sample_data and get_cmap are available utilities
from scripts.analysis_functions import random_sample_data
from utilities.plotting_tools import get_cmap

class PlotGenerator:
    def __init__(self, data_processor, base_directory, R=13.74, bootstrap=None,
                 pass_energies=None, overwrite=False):
        self.data_processor = data_processor
        self.base_directory = base_directory
        self.R = R
        self.bootstrap = bootstrap
        self.pass_energies = pass_energies
        self.overwrite = overwrite
        self.data = data_processor.data
        self.data_masked = data_processor.data_masked

    def generate_individual_ks_plots(self):
        data_masked = self.data_masked
        combinations = set(
            (entry['retardation'], entry['mid1_ratio'], entry['mid2_ratio'])
            for entry in data_masked
        )

        for retardation, mid1, mid2 in combinations:
            retardation_dir = os.path.join(self.base_directory, f"R{np.abs(retardation)}")
            os.makedirs(retardation_dir, exist_ok=True)

            pdf_filename = os.path.join(
                retardation_dir,
                f"ks_plot_R{retardation}_mid1_{mid1}_mid2_{mid2}.pdf"
            )

            if not self.overwrite and os.path.exists(pdf_filename):
                print(f"PDF already exists and overwrite is False. Skipping: {pdf_filename}")
                continue

            with PdfPages(pdf_filename) as pdf_pages:
                avg_ks_scores = plot_ks_score(
                    data_masked, [retardation], mid1, mid2, R=self.R, bootstrap=self.bootstrap, directory=None,
                    filename=None, pass_energies=self.pass_energies, pdf_pages=pdf_pages
                )

            print(f"Saved KS plot for retardation {retardation}, mid1 {mid1}, mid2 {mid2} "
                  f"to {pdf_filename} with ks score {avg_ks_scores}")

    def generate_ks_plots(self):
        data_masked = self.data_masked
        retardations = sorted(set(entry['retardation'] for entry in data_masked))

        for retardation in retardations:
            retardation_dir = os.path.join(self.base_directory, f"R{np.abs(retardation)}")
            os.makedirs(retardation_dir, exist_ok=True)

            pdf_filename = os.path.join(retardation_dir, f"ks_plots_R{retardation}.pdf")

            if not self.overwrite and os.path.exists(pdf_filename):
                print(f"PDF already exists and overwrite is False. Skipping: {pdf_filename}")
                continue

            with PdfPages(pdf_filename) as pdf_pages:
                mid1_mid2_combinations = set(
                    (entry['mid1_ratio'], entry['mid2_ratio']) for entry in data_masked if entry['retardation'] == retardation
                )

                for mid1, mid2 in mid1_mid2_combinations:
                    plot_ks_score(
                        data_masked, [retardation], mid1, mid2, R=self.R, bootstrap=self.bootstrap, directory=None,
                        filename=None, pass_energies=self.pass_energies, pdf_pages=pdf_pages
                    )

            print(f"Saved KS plots for retardation {retardation} to {pdf_filename}")

    def plot_relation(self, ax, ds, x_param, y_param, x_label, y_label, title=None,
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

    def plot_histogram(self, ax, parameter, x_label, title=None, nbins=50, selected_pairs=None):
        ds = self.data  # Use the data_masked from the data processor
        if selected_pairs:
            cmap = get_cmap(len(selected_pairs))
        else:
            cmap = get_cmap(len(ds))
        color_index = 0

        for i in range(len(ds)):
            entry = ds[i]
            retardation = entry['retardation']
            pass_energy = entry['pass_energy']
            mid1_ratio = entry['mid1_ratio']
            mid2_ratio = entry['mid2_ratio']
            pair = (retardation, pass_energy, mid1_ratio, mid2_ratio)

            if selected_pairs is None or pair in selected_pairs:
                collection_efficiency = entry.get('collection_efficiency', None)
                ks_score = entry.get('ks_score', None)
                data_to_plot = entry[parameter]

                label = f"R={retardation}, PE={pass_energy}"
                if collection_efficiency is not None:
                    label += f", CE={collection_efficiency}"
                if ks_score is not None:
                    label += f", KS={ks_score:.2f}"
                label += f", M1={mid1_ratio}, M2={mid2_ratio}"

                ax.hist(data_to_plot, bins=nbins, color=cmap(color_index),
                        edgecolor='black', alpha=0.5, label=label)
                color_index += 1

        if title:
            ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True)

if __name__ == '__main__':
    # Instantiate the DataProcessor with the custom data loader
    data_processor = DataProcessor(data_loader=custom_data_loader, compute_ranks=True)

    # Define ranges
    x_tof_range = (403.6, np.inf)
    retardation_range = (-1, -1)
    mid1_range = (0.08, 0.2)

    # Load data
    base_directory = r"C:\Users\proxi\Documents\coding\TOF_data"
    data_processor.load_data(
        base_dir=base_directory,
        x_tof_range=x_tof_range,
        retardation_range=retardation_range,
        #mid1_range=mid1_range,
        #mid2_range=mid1_range
    )

    # Define pass energies
    pass_energies = [1, 2, 5, 12, 18]

    # Initialize PlotGenerator
    plot_generator = PlotGenerator(
        data_processor=data_processor,
        base_directory=base_directory,
        R=13.74,
        bootstrap=10,
        pass_energies=pass_energies,
        overwrite=True
    )

    """# Generate KS plots
    plot_generator.generate_ks_plots()

    # Example usage of plot_relation
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_generator.plot_relation(
        ax=ax,
        ds=plot_generator.data_masked,
        x_param='x_param_name',  # Replace with actual parameter name
        y_param='y_param_name',  # Replace with actual parameter name
        x_label='X Axis Label',
        y_label='Y Axis Label',
        title='Plot Title',
        plot_log=False,
        retardation=(-10, 10),
        pass_energy=(0.3, 300),
        mid1_ratio=(0.08, 0.2),
        mid2_ratio=(0.08, 0.2),
        collection_efficiency=None,
        ks_score=None,
        verbose=False,
        sample_size=1000
    )
    plt.show()"""

    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes.flatten()

    # Define selected pairs if needed
    selected_pairs = [
        (-1, 1, 0.4, 0.08),
        (-1, 1, 0.3, 0.2),
        (-1, 1, 0.2, 0.3),
        (-1, 1, 0.11248, 0.1354)
    ]  # Replace with your specific pairs

    # Plot histograms using the plot_histogram method
    plot_generator.plot_histogram(
        ax=ax[0],
        parameter='tof_values',
        x_label='Time of Flight (um)',
        title='Histogram of Time of Flight',
        selected_pairs=selected_pairs  # Or use selected_pairs if you want to filter
    )

    plot_generator.plot_histogram(
        ax=ax[1],
        parameter='initial_elevation',
        x_label='Initial Elevation',
        title='Histogram of Initial Elevation',
        selected_pairs=selected_pairs
    )

    plot_generator.plot_histogram(
        ax=ax[2],
        parameter='x_tof',
        x_label='X position',
        title='Histogram of Final X position',
        selected_pairs=selected_pairs
    )

    plot_generator.plot_histogram(
        ax=ax[3],
        parameter='y_tof',
        x_label='Y position',
        title='Histogram of Final Y position',
        selected_pairs=selected_pairs
    )

    plt.tight_layout()
    plt.show()