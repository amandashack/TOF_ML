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
                    for key in data:
                        data[key] = data[key][mask]

                    # Calculate the collection efficiency
                    efficiency = len(data['initial_ke']) / start_len if start_len > 0 else 0

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
            for kinetic_energy, efficiency in energies.items():
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

    def plot_histogram(self, ax, parameter, x_label, title=None, nbins=50):
        cmap = get_cmap(len(self.data))
        for i in range(len(self.data)):
            ax.hist(self.data[i][parameter], bins=nbins, color=cmap(i),
                     edgecolor='black', alpha=0.5, label=f"Front Voltage={self.data[i]['retardation']}")
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
    amanda_filepath = dir_path + "/simion_output/positive_voltage"
    data_loader = DS_positive(filepath=amanda_filepath)
    data_loader.loadr(x_tof_range=(403.5, np.inf), y_tof_range=(-13.5, 13.5), min_pass=-10)