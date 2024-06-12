import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import re
import sys
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
import json
import xarray as xr
import arpys
from pyqtgraph.Qt import QtGui, QtWidgets
sys.path.insert(0, os.path.abspath('..'))
from loaders import DataStructure
from plotter import get_cmap


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, plot_func, *args, **kwargs):
        super().__init__()
        self.plot_func = plot_func
        self.args = args
        self.kwargs = kwargs
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        plot_widget = self.plot_func(*self.args, **self.kwargs)
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
        window = PlotWindow(arg.arpes.plot)
        app.exec_()
        window.close()


class DS_positive():
    def __init__(self):
        super().__init__()
        self.data = []
        self.data_masked = []
        self.collection_efficiency = {}
        self.ks_scores = {}

    def load_data(self, json_file, xtof_range, ytof_range, min_pass, retardation_range=None, mid1_range=None,
                  mid2_range=None):
        with open(json_file, 'r') as file:
            file_entries = json.load(file)

        for entry in file_entries:
            filename = entry['filename']  # assumes the json filename includes the full path
            retardation = entry['retardation']
            mid1 = entry['mid1_ratio']
            mid2 = entry['mid2_ratio']
            kinetic_energy = entry['kinetic_energy']

            # Apply filtering based on optional ranges
            if retardation_range and not (retardation_range[0] <= retardation <= retardation_range[1]):
                continue
            if mid1_range and not (mid1_range[0] <= mid1 <= mid1_range[1]):
                continue
            if mid2_range and not (mid2_range[0] <= mid2 <= mid2_range[1]):
                continue

            with h5py.File(filename, 'r') as f:
                initial_ke = f['data1']['initial_ke'][:]
                initial_elevation = f['data1']['initial_elevation'][:]
                x_tof = f['data1']['x'][:]
                y_tof = f['data1']['y'][:]
                tof_values = f['data1']['tof'][:]
                final_elevation = f['data1']['final_elevation'][:]
                final_ke = f['data1']['final_ke'][:]

                # Ensure all arrays are of the same length
                min_len = min(len(initial_ke), len(initial_elevation), len(x_tof), len(y_tof), len(tof_values),
                              len(final_elevation), len(final_ke))
                if min_len == 0:
                    continue

                initial_ke = initial_ke[:min_len]
                initial_elevation = initial_elevation[:min_len]
                x_tof = x_tof[:min_len]
                y_tof = y_tof[:min_len]
                tof_values = tof_values[:min_len]
                final_elevation = final_elevation[:min_len]
                final_ke = final_ke[:min_len]

                pass_energy = initial_ke + retardation

                data = {
                    'initial_ke': initial_ke,
                    'initial_elevation': initial_elevation,
                    'pass_energy': pass_energy,
                    'x_tof': x_tof,
                    'y_tof': y_tof,
                    'tof_values': tof_values,
                    'final_ke': final_ke,
                    'final_elevation': final_elevation,
                }

                start_len = len(data['initial_ke'])
                mask = self.create_mask(xtof_range, ytof_range, min_pass, data)

                # Ensure mask length matches data length
                if len(mask) != start_len:
                    print(f"Skipping file due to length mismatch: {filename}")
                    continue

                data_masked = {key: data[key][mask] for key in data}

                efficiency = len(data_masked['initial_ke']) / start_len if start_len > 0 else 0
                #if (kinetic_energy, mid1, mid2) in [(0, 0.4, 0.08), (0, 0.3, 0.2), (0, 0.2, 0.3), (0, 0.8, 0.1354), (0, 0.11248, 0.1354)]:
                #    print(mid1, mid2, kinetic_energy)
                #    ks_score = self.calculate_ks_score(data_masked['y_tof'], plot=True, print_ks=True)
                #else:
                ks_score = self.calculate_ks_score(data_masked['y_tof'])

                if (retardation, mid1, mid2) not in self.collection_efficiency:
                    self.collection_efficiency[(retardation, mid1, mid2)] = {}
                self.collection_efficiency[(retardation, mid1, mid2)][kinetic_energy] = efficiency

                self.data.append({
                    **data,
                    'retardation': retardation,
                    'mid1_ratio': mid1,
                    'mid2_ratio': mid2,
                    'kinetic_energy': kinetic_energy,
                    'collection_efficiency': efficiency,
                    'ks_score': ks_score
                })
                self.data_masked.append({
                    **data_masked,
                    'retardation': retardation,
                    'mid1_ratio': mid1,
                    'mid2_ratio': mid2,
                    'kinetic_energy': kinetic_energy,
                    'collection_efficiency': efficiency,
                    'ks_score': ks_score
                })

        # Store the retardation values for reference
        self.retardation = sorted(list(set([key[0] for key in self.collection_efficiency.keys()])))
        self.mid1_ratios = sorted(list(set([key[1] for key in self.collection_efficiency.keys()])))
        self.mid2_ratios = sorted(list(set([key[2] for key in self.collection_efficiency.keys()])))
        self.kinetic_energies = sorted(
            list(set([ke for subdict in self.collection_efficiency.values() for ke in subdict.keys()])))

    def create_mask(self, x_tof_range, y_tof_range, min_pass, data):
        xtof = np.asarray(data["x_tof"])[:].astype(float)
        ytof = np.abs(np.asarray(data["y_tof"])[:].astype(float))
        xmin_mask = xtof > x_tof_range[0]
        xmax_mask = xtof < x_tof_range[1]
        ymin_mask = ytof > y_tof_range[0]
        ymax_mask = ytof < y_tof_range[1]
        mask = xmin_mask & xmax_mask & ymin_mask & ymax_mask
        return mask

    def calculate_ks_score(self, y_pos, num_bootstrap=10, R=13.2, plot=False, print_ks=False):
        ks_scores = []
        num_points = len(y_pos)
        uniform_points = []
        final_points = []

        for _ in range(num_bootstrap):
            # Generate a uniform 2D radial distribution
            theta_uniform = np.random.uniform(0, 2 * np.pi, num_points)
            radius_uniform = np.random.uniform(0, R, num_points) ** 0.5
            x_uniform = radius_uniform * np.cos(theta_uniform)
            y_uniform = radius_uniform * np.sin(theta_uniform)

            # Convert final elevation to 2D coordinates
            theta_final = np.random.uniform(0, 2 * np.pi, num_points)
            radius_final = np.abs(y_pos) ** 0.5  # Assuming final_elevation is proportional to radius
            x_final = radius_final * np.cos(theta_final)
            y_final = radius_final * np.sin(theta_final)

            # Store points for plotting
            uniform_points.append((x_uniform, y_uniform))
            final_points.append((x_final, y_final))

            # Flatten the 2D arrays to 1D for KS test
            uniform_dist = np.hstack((x_uniform, y_uniform))
            final_dist = np.hstack((x_final, y_final))

            # Perform KS test
            ks_score, _ = ks_2samp(final_dist, uniform_dist)
            if print_ks:
                print(ks_score)
            ks_scores.append(ks_score)

        # Plotting the distributions if plot is True
        if plot:
            colors = plt.cm.viridis(np.linspace(0, 1, num_bootstrap))

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            for i in range(num_bootstrap):
                ax[0].scatter(uniform_points[i][0], uniform_points[i][1], alpha=0.5, color=colors[i])
                ax[1].scatter(final_points[i][0], final_points[i][1], alpha=0.5, color=colors[i])

            ax[0].set_title('Uniform Distribution')
            ax[0].set_xlabel('X')
            ax[0].set_ylabel('Y')
            ax[0].set_aspect('equal', 'box')

            ax[1].set_title('Final Elevation Distribution')
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('Y')
            ax[1].set_aspect('equal', 'box')

            plt.tight_layout()
            plt.show()

        return np.mean(ks_scores)

    def create_efficiency_xarray(self):
        # Create a 4D xarray for collection efficiency
        efficiency_data = np.full(
            (len(self.retardation), len(self.mid1_ratios), len(self.mid2_ratios), len(self.kinetic_energies)),
            np.nan
        )

        retardation_indices = {val: idx for idx, val in enumerate(self.retardation)}
        mid1_indices = {val: idx for idx, val in enumerate(self.mid1_ratios)}
        mid2_indices = {val: idx for idx, val in enumerate(self.mid2_ratios)}
        kinetic_energy_indices = {val: idx for idx, val in enumerate(self.kinetic_energies)}

        for (retardation, mid1, mid2), ke_dict in self.collection_efficiency.items():
            for kinetic_energy, efficiency in ke_dict.items():
                ridx = retardation_indices[retardation]
                mid1_idx = mid1_indices[mid1]
                mid2_idx = mid2_indices[mid2]
                ke_idx = kinetic_energy_indices[kinetic_energy]
                efficiency_data[ridx, mid1_idx, mid2_idx, ke_idx] = efficiency

        efficiency_xarray = xr.DataArray(
            efficiency_data,
            coords=[
                self.retardation,
                self.mid1_ratios,
                self.mid2_ratios,
                self.kinetic_energies
            ],
            dims=["retardation", "mid1_ratio", "mid2_ratio", "kinetic_energy"]
        )

        return efficiency_xarray

    def plot_relation(self, ax, ds, x_param, y_param, x_label, y_label, title=None,
                      plot_log=False, retardation=None, kinetic_energy=None, mid1_ratio=None, mid2_ratio=None,
                      collection_efficiency=None, ks_score=None, verbose=False):
        """
        Plots the specified parameters for the given dataset, with optional filtering by ranges.

        :param ax: Matplotlib axis object to plot on.
        :param ds: Data structure to plot, assumed to be a list of dictionaries.
        :param x_param: The parameter to plot on the x-axis.
        :param y_param: The parameter to plot on the y-axis.
        :param x_label: Label for the x-axis.
        :param y_label: Label for the y-axis.
        :param title: Title for the plot (optional).
        :param plot_log: Whether to plot the data on a log scale (default is False).
        :param retardation: Range for filtering retardation values (optional).
        :param kinetic_energy: Range for filtering kinetic energy values (optional).
        :param mid1_ratio: Range for filtering mid1_ratio values (optional).
        :param mid2_ratio: Range for filtering mid2_ratio values (optional).
        :param collection_efficiency: Range for filtering collection efficiency values (optional).
        :param ks_score: Range for filtering KS score values (optional).
        :param verbose: Whether to print the total number of data points plotted (default is False).
        """

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

        # Group data by (retardation, mid1_ratio, mid2_ratio)
        grouped_data = {}
        for item in data_to_plot:
            key = (item['retardation'], item['mid1_ratio'], item['mid2_ratio'])
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(item)

        # Define a colormap
        colors = get_cmap(len(grouped_data))

        for idx, (key, group) in enumerate(grouped_data.items()):
            color = colors(idx)
            for item in group:
                if not plot_log:
                    ax.scatter(item[x_param], item[y_param], alpha=0.6, color=color,
                               label=f"R={item['retardation']}, M1={item['mid1_ratio']}, M2={item['mid2_ratio']}")
                else:
                    ax.scatter(np.log2(item[x_param]), np.log2(item[y_param]), alpha=0.5, color=color,
                               label=f"R={item['retardation']}, M1={item['mid1_ratio']}, M2={item['mid2_ratio']}")

        # Remove duplicate labels in the legend
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

    def plot_heatmap(self):
        # Convert the collection_efficiency dictionary to a DataFrame
        data = []
        for retardation, energies in self.collection_efficiency.items():
            for kinetic_energy, efficiency in energies.items():
                data.append([retardation, kinetic_energy, efficiency])

        df = pd.DataFrame(data, columns=['Retardation', 'Kinetic Energy', 'Collection Efficiency'])

        # Pivot the DataFrame to get the correct format for the heatmap
        heatmap_data = df.pivot('Kinetic Energy', 'Retardation', 'Collection Efficiency')

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', cbar_kws={'label': 'Collection Efficiency'})
        plt.title('Heatmap of Collection Efficiency')
        plt.xlabel('Retardation')
        plt.ylabel('Kinetic Energy')
        plt.show()

    def plot_histogram(self, ax, ds, parameter, x_label, title=None, nbins=50, selected_pairs=None):
        """
        Plots histograms of the specified parameter for selected (retardation, kinetic_energy, mid1_ratio, mid2_ratio) pairs.

        :param ax: Matplotlib axis object to plot on.
        :param parameter: The parameter to plot the histogram for.
        :param x_label: Label for the x-axis.
        :param title: Title for the plot (optional).
        :param nbins: Number of bins for the histogram (default is 50).
        :param selected_pairs: List of (retardation, kinetic_energy, mid1_ratio, mid2_ratio) tuples to plot (default is None, meaning all data).
        """
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


if __name__ == '__main__':
    # Example usage
    # Assuming the files are located in a directory named 'data_files' in the current working directory.
    json_file = "simulation_data.json"
    data_loader = DS_positive()
    data_loader.load_data(json_file, (403.5, np.inf), (-20.5, 20.5), 0,
                          retardation_range=(1, 1))
    #print(data_loader.data[0])
    #data_loader.create_mask(x_tof_range=(403.5, np.inf), y_tof_range=(-13.5, 13.5), min_pass=-10)

    """fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Create a figure with multiple subplots
    ax = axes.flatten()
    selected = [(1, 0, 0.4, 0.08), (1, 0, 0.3, 0.2), (1, 0, 0.2, 0.3), (1, 0, 0.8, 0.1354), (1, 0, 0.11248, 0.1354)]  # Replace with your specific pairs
    data_loader.plot_histogram(ax[0], data_loader.data_masked, parameter='initial_elevation', x_label='Initial Elevation',
                               title='Histogram of Initial Elevation', selected_pairs=selected)
    data_loader.plot_histogram(ax[1], data_loader.data_masked, parameter='final_elevation', x_label='Final Elevation',
                               title='Histogram of Final Elevation', selected_pairs=selected)
    data_loader.plot_histogram(ax[2], data_loader.data_masked, parameter='x_tof', x_label='X position',
                               title='Histogram of Final X position', selected_pairs=selected)
    data_loader.plot_histogram(ax[3], data_loader.data_masked, parameter='y_tof', x_label='Y position',
                               title='Histogram of Final Y position', selected_pairs=selected)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define ranges for filtering
    retardation_range = (1, 1)
    kinetic_energy_range = (0, 5)
    mid1_ratio_range = (0.08, 0.4)
    mid2_ratio_range = (0.08, 0.4)
    #collection_efficiency_range = (0.3, 1)
    #ks_score_range = (0, 0.2)

    # Call the function with filtering ranges
    data_loader.plot_relation(
        ax=ax,
        ds=data_loader.data,
        x_param='pass_energy',
        y_param='tof_values',
        x_label='Pass Energy (eV)',
        y_label='Time of Flight',
        title='TOF vs. Pass Energy',
        plot_log=False,
        retardation=retardation_range,
        kinetic_energy=kinetic_energy_range,
        mid1_ratio=mid1_ratio_range,
        mid2_ratio=mid2_ratio_range,
        #collection_efficiency=collection_efficiency_range,
        #ks_score=ks_score_range,
        verbose=True
    )

    plt.show()"""
    #data_loader.plot_heatmap()
    ex = data_loader.create_efficiency_xarray()
    plot_imagetool(ex.sel({'retardation': 1}))
    """fig, ax = plt.subplots()
    ex.sel({'kinetic_energy': 0}).plot(ax=ax)
    plt.xlabel('Blade 22 ratio')
    plt.ylabel('Blade 25 ratio')
    plt.title('Collection efficiency')
    plt.legend()
    plt.show()"""
