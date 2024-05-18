import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import re
import sys
import pandas as pd
import seaborn as sns
sys.path.insert(0, os.path.abspath('..'))
from loaders import DataStructure
from plotter import get_cmap


class DS_positive(DataStructure):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.data_masked = []

    def loadr(self, x_tof_range, y_tof_range, min_pass, retardation_value=None):
        """
        Loads data from all HDF5 files in the specified directory that match the naming pattern.
        Assumes the HDF5 files have specific datasets within them.
        """
        fp = os.path.abspath(self.filepath)
        self.collection_efficiency = {}  # Dictionary to store collection efficiency by front_voltage and kinetic_energy
        self.data = []  # Reset self.data
        self.retardation = []  # Reset self.retardation

        for file in os.listdir(fp):
            if file.endswith("h5"):
                path1 = os.path.join(fp, file)
                try:
                    # Extract front_voltage and kinetic_energy from the file name
                    match = re.findall(r'_R0_(\d+)_(\d+)', file)
                    if match:
                        front_voltage, kinetic_energy = map(int, match[0])
                    else:
                        continue  # Skip files that do not match the pattern
                except Exception as e:
                    print(e)
                    continue

                with h5py.File(path1, 'r') as f:
                    # Extracting the necessary datasets
                    initial_ke = f['data1']['initial_ke'][:]
                    initial_elevation = f['data1']['initial_elevation'][:]
                    x_tof = f['data1']['x'][:]
                    y_tof = f['data1']['y'][:]
                    tof_values = f['data1']['tof'][:]
                    final_elevation = f['data1']['final_elevation'][:]
                    final_ke = f['data1']['final_ke'][:]

                    # Calculate the pass energy
                    pass_energy = initial_ke + front_voltage

                    # Store the data in a structured way
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

                    # Calculate the initial length before masking
                    start_len = len(data['initial_ke'])

                    # Apply the mask
                    mask = self.create_mask(x_tof_range, y_tof_range, min_pass, data)
                    data_masked = data.copy()
                    for key in data:
                        data_masked[key] = data[key][mask]

                    # Calculate the collection efficiency
                    efficiency = len(data_masked['initial_ke']) / start_len #if start_len > 0 else 0

                    # Store the collection efficiency
                    if front_voltage not in self.collection_efficiency:
                        self.collection_efficiency[front_voltage] = {}
                    self.collection_efficiency[front_voltage][kinetic_energy] = efficiency

                    # Append the data and efficiency to self.data
                    self.data.append({
                        **data,
                        'retardation': front_voltage,
                        'collection_efficiency': efficiency,
                    })
                    self.data_masked.append({
                        **data_masked,
                        'retardation': front_voltage,
                        'collection_efficiency': efficiency,
                    })

        # Store the front_voltage values for reference
        self.retardation = list(self.collection_efficiency.keys())

    def create_mask(self, x_tof_range, y_tof_range, min_pass, data):
        xtof = np.asarray(data["x_tof"])[:].astype(float)
        ytof = np.abs(np.asarray(data["y_tof"])[:].astype(float))
        #pass_en = np.asarray(data["pass_energy"])[:].astype(float)
        xmin_mask = xtof > x_tof_range[0]
        xmax_mask = xtof < x_tof_range[1]
        ymin_mask = ytof > y_tof_range[0]
        ymax_mask = ytof < y_tof_range[1]
        #pass_mask = pass_en > min_pass
        mask = xmin_mask & xmax_mask & ymin_mask & ymax_mask #& pass_mask
        return mask

    def plot_relation(self, ax, ds, x_param, y_param, x_label, y_label, title=None,
                      plot_log=False, specific_retardations=None, verbose=False):
        """
        Plots TOF versus Pass Energy for all or specified retardation values.
        :param ds: data structure you would like to plot, assumed to be a list of dictionaries where
        each list corresponds to a different retardation
        :param specific_retardations: List of retardation values to plot. If None, plots for all retardation values.
        """
        if specific_retardations is not None:
            data_to_plot = [item for item in ds if item['retardation'] in specific_retardations]
        else:
            data_to_plot = ds
        for item in data_to_plot:
            if not plot_log:
                ax.scatter(item[x_param], item[y_param], alpha=0.6, label=f"R={item['retardation']}")
            else:
                ax.scatter(np.log2(item[x_param]), np.log2(item[y_param]), alpha=0.5, label=f"R={item['retardation']}")
        if verbose:
            tot = sum([len(d[y_param]) for d in ds])
            print(tot)
        if title:
            ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True)

    def plot_heatmap(self):
        # Convert the collection_efficiency dictionary to a DataFrame
        data = []
        for front_voltage, energies in self.collection_efficiency.items():
            print(front_voltage, energies)
            for kinetic_energy, efficiency in energies.items():
                print(kinetic_energy, efficiency)
                data.append([front_voltage, kinetic_energy, efficiency])

        df = pd.DataFrame(data, columns=['Front Voltage', 'Kinetic Energy', 'Collection Efficiency'])

        # Pivot the DataFrame to get the correct format for the heatmap
        heatmap_data = df.pivot('Kinetic Energy', 'Front Voltage', 'Collection Efficiency')

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', cbar_kws={'label': 'Collection Efficiency'})
        plt.title('Heatmap of Collection Efficiency')
        plt.xlabel('Front Voltage')
        plt.ylabel('Kinetic Energy')
        plt.show()

    def plot_histogram(self, ax, ds, parameter, x_label, title=None, nbins=50, selected_pairs=None):
        """
        Plots histograms of the specified parameter for selected (front_voltage, kinetic_energy) pairs.

        :param ax: Matplotlib axis object to plot on.
        :param parameter: The parameter to plot the histogram for.
        :param x_label: Label for the x-axis.
        :param title: Title for the plot (optional).
        :param nbins: Number of bins for the histogram (default is 50).
        :param selected_pairs: List of (front_voltage, kinetic_energy) tuples to plot (default is None, meaning all data).
        """
        cmap = get_cmap(len(selected_pairs))
        color_index = 0

        for i in range(len(ds)):
            front_voltage = ds[i]['retardation']
            try:
                kinetic_energy = ds[i]['initial_ke'][0]
            except IndexError:
                print("This combination was completely masked out: ",
                      front_voltage, self.data[i]['initial_ke'][0])
                continue

            if selected_pairs is None or (front_voltage, kinetic_energy) in selected_pairs:
                ax.hist(ds[i][parameter], bins=nbins, color=cmap(color_index),
                        edgecolor='black', alpha=0.5, label=f"FV={front_voltage}, KE={kinetic_energy}")
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
    dir_path = "C:/Users/proxi/Documents/coding/TOF_ML/simulations/TOF_simulation"
    amanda_filepath = dir_path + "/simion_output/positive_voltage/nonNM"
    data_loader = DS_positive(filepath=amanda_filepath)
    data_loader.loadr(x_tof_range=(0, np.inf), y_tof_range=(-15.5, 15.5), min_pass=-10)
    #print(data_loader.data[0])
    #data_loader.create_mask(x_tof_range=(403.5, np.inf), y_tof_range=(-13.5, 13.5), min_pass=-10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Create a figure with multiple subplots
    ax = axes.flatten()
    selected_pairs = [(1, 1), (1, 2), (5, 3), (8, 6), (8, 10), (8, 18)]  # Replace with your specific pairs
    data_loader.plot_histogram(ax[0], data_loader.data_masked, parameter='initial_ke', x_label='Initial KE', title='Histogram of Initial KE',
                          selected_pairs=selected_pairs)
    data_loader.plot_histogram(ax[1], data_loader.data_masked, parameter='final_ke', x_label='Final KE', title='Histogram of Final KE',
                               selected_pairs=selected_pairs)
    data_loader.plot_histogram(ax[2], data_loader.data_masked, parameter='x_tof', x_label='X position', title='Histogram of Final X position',
                               selected_pairs=selected_pairs)
    data_loader.plot_histogram(ax[3], data_loader.data_masked, parameter='y_tof', x_label='Y position', title='Histogram of Final Y position',
                               selected_pairs=selected_pairs)
    #data_loader.plot_histogram(ax[0], "pass_energy", "Pass Energy (eV)", title=None, nbins=50)
    #data_loader.plot_histogram(ax[1], "x_tof", "x position", title=None, nbins=50)
    #data_loader.plot_histogram(ax[2], "y_tof", "y position", title=None, nbins=50)
    #data_loader.plot_histogram(ax[3], "initial_elevation", "initial elevation", title=None, nbins=50)
    plt.show()
    data_loader.plot_heatmap()
