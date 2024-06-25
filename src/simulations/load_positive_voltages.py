import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import sys
import json
import xarray as xr
sys.path.insert(0, os.path.abspath('..'))
from loaders.load_and_save import save_xarray, load_xarray, save_to_h5, load_from_h5
from utilities.plotting_tools import plot_imagetool, plot_relation, plot_heatmap, plot_histogram, plot_energy_resolution
from utilities.mask_data import create_mask
from utilities.calculation_tools import calculate_ks_score, normalize_3D


class DS_positive():
    def __init__(self):
        super().__init__()
        self.data = []
        self.data_masked = []
        self.collection_efficiency = {}
        self.ks_scores = {}
        self.retardation = []
        self.mid1_ratios = []
        self.mid2_ratios = []
        self.kinetic_energies = []

    def load_data(self, json_file, xtof_range=None, ytof_range=None, retardation_range=None,
                  mid1_range=None, mid2_range=None, overwrite=False):
        with open(json_file, 'r') as file:
            file_entries = json.load(file)

        combined_files = [entry for entry in file_entries if entry.get('combined', False)]

        for entry in combined_files:
            filename = entry['filename']
            retardation = entry['retardation']

            if retardation_range and not (retardation_range[0] <= retardation <= retardation_range[1]):
                continue

            with h5py.File(filename, 'r+') as f:
                for group_name in f.keys():
                    grp = f[group_name]
                    mid1 = grp.attrs['mid1_ratio']
                    mid2 = grp.attrs['mid2_ratio']
                    kinetic_energy = grp.attrs['kinetic_energy']
                    if kinetic_energy == 0:
                        kinetic_energy = 0.1

                    if mid1_range and not (mid1_range[0] <= mid1 <= mid1_range[1]):
                        continue
                    if mid2_range and not (mid2_range[0] <= mid2 <= mid2_range[1]):
                        continue

                    initial_ke = grp['initial_ke'][:]
                    initial_elevation = grp['initial_elevation'][:]
                    x_tof = grp['x_tof'][:]
                    y_tof = grp['y_tof'][:]
                    tof_values = grp['tof_values'][:]
                    final_elevation = grp['final_elevation'][:]
                    final_ke = grp['final_ke'][:]

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

                    if 'collection_efficiency' in grp.attrs and 'ks_score' in grp.attrs and 'mask' in grp and not overwrite:
                        efficiency = grp.attrs['collection_efficiency']
                        ks_score = grp.attrs['ks_score']
                        mask = grp['mask'][:]
                        data_masked = {key: data[key][mask] for key in data}
                    else:
                        if xtof_range is None or ytof_range is None:
                            raise ValueError("xtof_range, ytof_range, and min_pass must be provided if not already recorded in the H5 file")

                        start_len = len(data['initial_ke'])
                        mask = create_mask(data, xtof_range, ytof_range)

                        if len(mask) != start_len:
                            print(f"Skipping file due to length mismatch: {filename}")
                            continue

                        data_masked = {key: data[key][mask] for key in data}

                        efficiency = len(data_masked['initial_ke']) / start_len if start_len > 0 else 0
                        ks_score = calculate_ks_score(data_masked['y_tof'])

                        grp.attrs['collection_efficiency'] = efficiency
                        grp.attrs['ks_score'] = ks_score

                        if 'mask' in grp and overwrite:
                            del grp['mask']
                        grp.create_dataset('mask', data=mask)

                    if (retardation, mid1, mid2) not in self.collection_efficiency:
                        self.collection_efficiency[(retardation, mid1, mid2)] = {}
                    self.collection_efficiency[(retardation, mid1, mid2)][kinetic_energy] = efficiency

                    if (retardation, mid1, mid2) not in self.ks_scores:
                        self.ks_scores[(retardation, mid1, mid2)] = {}
                    self.ks_scores[(retardation, mid1, mid2)][kinetic_energy] = ks_score

                    self.data.append({
                        **data,
                        'retardation': retardation,
                        'mid1_ratio': mid1,
                        'mid2_ratio': mid2,
                        'kinetic_energy': kinetic_energy,
                        'collection_efficiency': efficiency,
                        'ks_score': ks_score,
                        'mask': mask
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

        self.retardation = sorted(list(set([key[0] for key in self.collection_efficiency.keys()])))
        self.mid1_ratios = sorted(list(set([key[1] for key in self.collection_efficiency.keys()])))
        self.mid2_ratios = sorted(list(set([key[2] for key in self.collection_efficiency.keys()])))
        self.kinetic_energies = sorted(list(set([ke for subdict in self.collection_efficiency.values()
                                          for ke in subdict.keys()])))

    def create_efficiency_xarray(self):
        r_coord = list(range(np.min(self.retardation), np.max(self.retardation)+1))
        efficiency_data = np.full(
            (len(r_coord), len(self.mid1_ratios), len(self.mid2_ratios), len(self.kinetic_energies)),
            np.nan
        )

        retardation_indices = {val: idx for idx, val in enumerate(r_coord)}
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
                r_coord,
                self.mid1_ratios,
                self.mid2_ratios,
                self.kinetic_energies
            ],
            dims=["retardation", "mid1_ratio", "mid2_ratio", "kinetic_energy"]
        )

        return efficiency_xarray

    def create_ks_score_xarray(self):
        r_coord = list(range(np.min(self.retardation), np.max(self.retardation)+1))
        ks_data = np.full(
            (len(r_coord), len(self.mid1_ratios), len(self.mid2_ratios), len(self.kinetic_energies)),
            np.nan
        )

        retardation_indices = {val: idx for idx, val in enumerate(r_coord)}
        mid1_indices = {val: idx for idx, val in enumerate(self.mid1_ratios)}
        mid2_indices = {val: idx for idx, val in enumerate(self.mid2_ratios)}
        kinetic_energy_indices = {val: idx for idx, val in enumerate(self.kinetic_energies)}

        for (retardation, mid1, mid2), ke_dict in self.ks_scores.items():
            for kinetic_energy, ks_score in ke_dict.items():
                ridx = retardation_indices[retardation]
                mid1_idx = mid1_indices[mid1]
                mid2_idx = mid2_indices[mid2]
                ke_idx = kinetic_energy_indices[kinetic_energy]
                ks_data[ridx, mid1_idx, mid2_idx, ke_idx] = ks_score

        efficiency_xarray = xr.DataArray(
            ks_data,
            coords=[
                r_coord,
                self.mid1_ratios,
                self.mid2_ratios,
                self.kinetic_energies
            ],
            dims=["retardation", "mid1_ratio", "mid2_ratio", "kinetic_energy"]
        )

        return efficiency_xarray

    def calculate_energy_resolution(self):
        gradients = {}
        for (retardation, mid1, mid2) in self.collection_efficiency.keys():
            kinetic_energies = sorted(self.collection_efficiency[(retardation, mid1, mid2)].keys())
            avg_tof_values = [
                np.mean([np.log(item['tof_values']) for item in self.data_masked
                         if item['retardation'] == retardation and
                         item['mid1_ratio'] == mid1 and
                         item['mid2_ratio'] == mid2 and
                         item['kinetic_energy'] == ke])
                for ke in kinetic_energies
            ]
            if len(avg_tof_values) < 21:
                print(avg_tof_values, kinetic_energies, retardation, mid1, mid2)
            var_tof_values = [
                np.var([item['tof_values'] for item in self.data_masked
                         if item['retardation'] == retardation and
                         item['mid1_ratio'] == mid1 and
                         item['mid2_ratio'] == mid2 and
                         item['kinetic_energy'] == ke])
                for ke in kinetic_energies
            ]

            if len(kinetic_energies) > 1:
                gradient = np.gradient(avg_tof_values, np.log(kinetic_energies))
                error = []

                # Propagate errors
                for i in range(len(gradient)):
                    if i == 0:
                        # Boundary case: forward difference
                        error.append(var_tof_values[1] + var_tof_values[0])
                    elif i == len(gradient) - 1:
                        # Boundary case: backward difference
                        error.append(var_tof_values[-1] + var_tof_values[-2])
                    else:
                        # Central difference
                        error.append((var_tof_values[i + 1] + var_tof_values[i - 1]) / 2)

                gradients[(retardation, mid1, mid2)] = [avg_tof_values, kinetic_energies, gradient, error]

        return gradients

    def create_energy_resolution_xarray(self, resolution_dict):
        r_coord = list(range(np.min(self.retardation), np.max(self.retardation)+1))
        resolution_data = np.full(
            (len(r_coord), len(self.mid1_ratios), len(self.mid2_ratios), len(self.kinetic_energies)),
            np.nan
        )
        retardation_indices = {val: idx for idx, val in enumerate(r_coord)}
        mid1_indices = {val: idx for idx, val in enumerate(self.mid1_ratios)}
        mid2_indices = {val: idx for idx, val in enumerate(self.mid2_ratios)}

        for (retardation, mid1, mid2), gradient in resolution_dict.items():
            ridx = retardation_indices[retardation]
            mid1_idx = mid1_indices[mid1]
            mid2_idx = mid2_indices[mid2]
            resolution_data[ridx, mid1_idx, mid2_idx, :] = gradient[2]

        resolution_xarray = xr.DataArray(
            resolution_data,
            coords=[
                r_coord,
                self.mid1_ratios,
                self.mid2_ratios,
                self.kinetic_energies
            ],
            dims=["retardation", "mid1_ratio", "mid2_ratio", "kinetic_energy"]
        )

        return resolution_xarray

    def create_combined_array(self):
        combined_data = []

        for entry in self.data:
            initial_ke = entry['pass_energy']
            elevation = entry['initial_elevation']
            retardation = entry['retardation']
            mid1_ratio = entry['mid1_ratio']
            mid2_ratio = entry['mid2_ratio']
            tof = entry['tof_values']
            y_tof = entry['y_tof']
            mask = entry['mask']

            retardation_array = np.full_like(initial_ke, retardation)
            mid1_ratio_array = np.full_like(initial_ke, mid1_ratio)
            mid2_ratio_array = np.full_like(initial_ke, mid2_ratio)

            combined_entry = np.column_stack([
                initial_ke,
                elevation,
                retardation_array,
                mid1_ratio_array,
                mid2_ratio_array,
                tof,
                y_tof,
                mask
            ])

            combined_data.append(combined_entry)

        combined_array = np.vstack(combined_data)
        return combined_array



if __name__ == '__main__':
    # Example usage
    # Assuming the files are located in a directory named 'data_files' in the current working directory.
    json_file = "simulation_data.json"
    data_loader = DS_positive()
    data_loader.load_data('simulation_data.json', xtof_range=(403.6, np.inf), ytof_range=(-13.74, 13.74),
                          retardation_range=(-10, 10), overwrite=False)#, mid1_range=(0.11248, 0.11248), mid2_range=(0.1354, 0.1354), overwrite=False)
    # Create the combined array
    combined_array = data_loader.create_combined_array()

    # Save the combined array to an HDF5 file
    save_to_h5(combined_array, 'combined_data.h5')
    #ex = data_loader.create_efficiency_xarray()
    #plot_imagetool(ex.sel({'kinetic_energy': 0.1}))
    #save_xarray(ex, r"C:\Users\proxi\Documents\coding\TOF_ML\simulations\TOF_simulation\simion_output\collection_efficiency",
    #            "collection_efficiency")
    #ks = data_loader.create_ks_score_xarray()
    #save_xarray(ks,
    #            r"C:\Users\proxi\Documents\coding\TOF_ML\simulations\TOF_simulation\simion_output\collection_efficiency",
    #            "ks_score")
    #resolution_dict = data_loader.calculate_energy_resolution()
    #plot_energy_resolution(resolution_dict, retardation=1, mid1_range=(0.2, 0.4), mid2_range=(0.2, 0.4))

    #resolution_xarray = data_loader.create_energy_resolution_xarray(resolution_dict)
    #save_xarray(resolution_xarray,
    #            r"C:\Users\proxi\Documents\coding\TOF_ML\simulations\TOF_simulation\simion_output\collection_efficiency",
    #            "resolution")
    #efficiency_array = load_xarray(r"C:\Users\proxi\Documents\coding\TOF_ML\simulations\TOF_simulation\simion_output\collection_efficiency",
    #            "collection_efficiency")
    #plot_imagetool(efficiency_array.sel({'kinetic_energy': 0}))
    #print(data_loader.data[0])
    #data_loader.create_mask(x_tof_range=(403.5, np.inf), y_tof_range=(-13.5, 13.5), min_pass=-10)

    #efficiency_array = load_xarray(
    #    r"C:\Users\proxi\Documents\coding\TOF_ML\simulations\TOF_simulation\simion_output\collection_efficiency",
    #    "collection_efficiency")
    #plot_imagetool(efficiency_array.sel({'kinetic_energy': 0, 'retardation': 1}))
    """fig, ax = plt.subplots()
    efficiency_array.sel({'kinetic_energy': 0}).plot(ax=ax)
    plt.xlabel('Blade 22 ratio')
    plt.ylabel('Blade 25 ratio')
    plt.title('Collection efficiency')
    plt.legend()
    plt.show()"""

    """fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Create a figure with multiple subplots
    ax = axes.flatten()
    selected = [(1, 0, 0.4, 0.08), (1, 0, 0.3, 0.2), (1, 0, 0.2, 0.3), (1, 0, 0.8, 0.1354), (1, 0, 0.11248, 0.1354)]  # Replace with your specific pairs
    plot_histogram(ax[0], data_loader.data_masked, parameter='initial_elevation', x_label='Initial Elevation',
                               title='Histogram of Initial Elevation', selected_pairs=None)
    plot_histogram(ax[1], data_loader.data_masked, parameter='final_elevation', x_label='Final Elevation',
                               title='Histogram of Final Elevation', selected_pairs=None)
    plot_histogram(ax[2], data_loader.data_masked, parameter='x_tof', x_label='X position',
                               title='Histogram of Final X position', selected_pairs=None)
    plot_histogram(ax[3], data_loader.data_masked, parameter='y_tof', x_label='Y position',
                               title='Histogram of Final Y position', selected_pairs=None)
    plt.show()"""

    """fig, ax = plt.subplots(figsize=(10, 6))

    # Define ranges for filtering
    retardation_range = (1, 1)
    kinetic_energy_range = (0, 5)
    mid1_ratio_range = (0.08, 0.4)
    mid2_ratio_range = (0.08, 0.4)
    #collection_efficiency_range = (0.3, 1)
    #ks_score_range = (0, 0.2)

    # Call the function with filtering ranges
    plot_relation(
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
