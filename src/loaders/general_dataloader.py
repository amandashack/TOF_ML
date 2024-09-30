import numpy as np
import xarray as xr
import h5py
import os
import re
from .load_h5_files_to_array import parse_filename
from utilities.calculation_tools import calculate_ks_score, normalize_3D
from utilities.mask_data import create_mask
from utilities.plotting_tools import plot_ks_score, plot_imagetool

class DataProcessor:
    def __init__(self, data_loader, compute_ranks=True):
        self.data_loader = data_loader
        self.compute_ranks = compute_ranks
        self.data = []
        self.data_masked = []  # Initialize masked data list
        self.retardation = []
        self.mid1_ratios = []
        self.mid2_ratios = []
        self.pass_energies = []
        self.collection_efficiency = {}
        self.ks_scores = {}

    def load_data(self, *args, **kwargs):
        loaded_data = self.data_loader(*args, **kwargs)
        self.process_loaded_data(loaded_data)

    def process_loaded_data(self, loaded_data):
        for entry in loaded_data:
            # Process each entry and update internal state
            self.data.append(entry)
            self.data_masked.append(entry['data_masked'])
            retardation = entry['retardation']
            mid1 = entry['mid1_ratio']
            mid2 = entry['mid2_ratio']
            pass_energy = entry['pass_energy']
            efficiency = entry.get('collection_efficiency', 0)
            ks_score = entry.get('ks_score', 0)

            # Update unique values
            self.retardation.append(retardation)
            self.mid1_ratios.append(mid1)
            self.mid2_ratios.append(mid2)
            self.pass_energies.append(pass_energy)

            # Update efficiency and ks_scores dictionaries
            key = (retardation, mid1, mid2)
            if key not in self.collection_efficiency:
                self.collection_efficiency[key] = {}
            self.collection_efficiency[key][pass_energy] = efficiency

            if key not in self.ks_scores:
                self.ks_scores[key] = {}
            self.ks_scores[key][pass_energy] = ks_score

        # Remove duplicates
        self.retardation = sorted(set(self.retardation))
        self.mid1_ratios = sorted(set(self.mid1_ratios))
        self.mid2_ratios = sorted(set(self.mid2_ratios))
        self.pass_energies = sorted(set(self.pass_energies))
        print(self.pass_energies, self.retardation, self.mid2_ratios, self.mid1_ratios)

        # Compute ranks if enabled
        if self.compute_ranks:
            self.compute_ranks_method()

    def set_compute_ranks(self, r):
        self.compute_ranks = r

    def compute_ranks_method(self):
        # For each (retardation, pass_energy), compute normalized ranks
        # Create a mapping from (retardation, pass_energy) to entries
        data_grouped = {}
        for entry in self.data:
            retardation = entry['retardation']
            pass_energy = entry['pass_energy']
            key = (retardation, pass_energy)
            if key not in data_grouped:
                data_grouped[key] = []
            data_grouped[key].append(entry)

        # Compute normalized ranks for collection efficiency and KS score
        for key, entries in data_grouped.items():
            # Collection Efficiency
            ce_values = [entry['collection_efficiency'] for entry in entries]
            ce_min = min(ce_values)
            ce_max = max(ce_values)
            ce_range = ce_max - ce_min if ce_max != ce_min else 1  # Avoid division by zero
            for entry in entries:
                ce = entry['collection_efficiency']
                # Normalize CE: Higher CE gets rank closer to 1
                entry['ce_rank'] = (ce - ce_min) / ce_range

            # KS Score
            ks_values = [entry['ks_score'] for entry in entries]
            ks_min = min(ks_values)
            ks_max = max(ks_values)
            ks_range = ks_max - ks_min if ks_max != ks_min else 1  # Avoid division by zero
            for entry in entries:
                ks = entry['ks_score']
                # Normalize KS score: Lower KS gets rank closer to 1
                entry['ks_score_rank'] = (ks_max - ks) / ks_range

        # Assign ranks to data_masked entries
        for entry in self.data_masked:
            # Find the corresponding unmasked entry
            matching_entries = [e for e in self.data if e['retardation'] == entry['retardation'] and
                                e['pass_energy'] == entry['pass_energy'] and
                                e['mid1_ratio'] == entry['mid1_ratio'] and
                                e['mid2_ratio'] == entry['mid2_ratio']]
            if matching_entries:
                entry['ce_rank'] = matching_entries[0].get('ce_rank', None)
                entry['ks_score_rank'] = matching_entries[0].get('ks_score_rank', None)

    def create_efficiency_xarray(self):
        r_coord = list(range(np.min(self.retardation), np.max(self.retardation) + 1))
        efficiency_data = np.full(
            (len(r_coord), len(self.mid1_ratios), len(self.mid2_ratios), len(self.pass_energies)),
            np.nan
        )

        retardation_indices = {val: idx for idx, val in enumerate(r_coord)}
        mid1_indices = {val: idx for idx, val in enumerate(self.mid1_ratios)}
        mid2_indices = {val: idx for idx, val in enumerate(self.mid2_ratios)}
        pass_energy_indices = {val: idx for idx, val in enumerate(self.pass_energies)}

        for (retardation, mid1, mid2), ke_dict in self.collection_efficiency.items():
            for pass_energy, efficiency in ke_dict.items():
                ridx = retardation_indices[retardation]
                mid1_idx = mid1_indices[mid1]
                mid2_idx = mid2_indices[mid2]
                pe_idx = pass_energy_indices[pass_energy]
                efficiency_data[ridx, mid1_idx, mid2_idx, pe_idx] = efficiency

        efficiency_xarray = xr.DataArray(
            efficiency_data,
            coords=[
                r_coord,
                self.mid1_ratios,
                self.mid2_ratios,
                self.pass_energies
            ],
            dims=["retardation", "mid1_ratio", "mid2_ratio", "pass_energy"]
        )

        return efficiency_xarray

    def create_ks_score_xarray(self):
        r_coord = list(range(np.min(self.retardation), np.max(self.retardation) + 1))
        efficiency_data = np.full(
            (len(r_coord), len(self.mid1_ratios), len(self.mid2_ratios), len(self.pass_energies)),
            np.nan
        )

        retardation_indices = {val: idx for idx, val in enumerate(r_coord)}
        mid1_indices = {val: idx for idx, val in enumerate(self.mid1_ratios)}
        mid2_indices = {val: idx for idx, val in enumerate(self.mid2_ratios)}
        pass_energy_indices = {val: idx for idx, val in enumerate(self.pass_energies)}

        for (retardation, mid1, mid2), pe_dict in self.ks_scores.items():
            for pass_energy, ks_score in pe_dict.items():
                ridx = retardation_indices[retardation]
                mid1_idx = mid1_indices[mid1]
                mid2_idx = mid2_indices[mid2]
                pe_idx = pass_energy_indices[pass_energy]
                efficiency_data[ridx, mid1_idx, mid2_idx, pe_idx] = ks_score

        efficiency_xarray = xr.DataArray(
            efficiency_data,
            coords=[
                r_coord,
                self.mid1_ratios,
                self.mid2_ratios,
                self.pass_energies
            ],
            dims=["retardation", "mid1_ratio", "mid2_ratio", "pass_energy"]
        )

        return efficiency_xarray

    def calculate_energy_resolution(self):
        gradients = {}
        r1 = 99999
        for (retardation, mid1, mid2) in self.collection_efficiency.keys():
            if r1 != retardation:
                # used for printing progress
                r1 = retardation
                print(retardation)
            pass_energies = sorted(self.collection_efficiency[(retardation, mid1, mid2)].keys())
            avg_tof_values = [
                np.mean([item['tof_values'] for item in self.data_masked
                         if item['retardation'] == retardation and
                         item['mid1_ratio'] == mid1 and
                         item['mid2_ratio'] == mid2 and
                         item['pass_energy'] == pe])
                for pe in pass_energies
            ]
            if len(avg_tof_values) < 21:
                print(avg_tof_values, pass_energies, retardation, mid1, mid2)
            var_tof_values = [
                np.var([item['tof_values'] for item in self.data_masked
                        if item['retardation'] == retardation and
                        item['mid1_ratio'] == mid1 and
                        item['mid2_ratio'] == mid2 and
                        item['pass_energy'] == pe])
                for pe in pass_energies
            ]

            if len(pass_energies) > 1:
                gradient = np.gradient(np.array(avg_tof_values), np.array(pass_energies))
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

                gradients[(retardation, mid1, mid2)] = [avg_tof_values, gradient, error]
        print("Finished making gradients dict")
        return gradients

    def create_energy_resolution_xarray(self, resolution_dict):
        r_coord = list(range(np.min(self.retardation), np.max(self.retardation) + 1))
        resolution_data = np.full(
            (len(r_coord), len(self.mid1_ratios), len(self.mid2_ratios), len(self.pass_energies)),
            np.nan
        )
        retardation_indices = {val: idx for idx, val in enumerate(r_coord)}
        mid1_indices = {val: idx for idx, val in enumerate(self.mid1_ratios)}
        mid2_indices = {val: idx for idx, val in enumerate(self.mid2_ratios)}

        for (retardation, mid1, mid2), gradient in resolution_dict.items():
            ridx = retardation_indices[retardation]
            mid1_idx = mid1_indices[mid1]
            mid2_idx = mid2_indices[mid2]
            resolution_data[ridx, mid1_idx, mid2_idx, :] = gradient[1]

        resolution_xarray = xr.DataArray(
            resolution_data,
            coords=[
                r_coord,
                self.mid1_ratios,
                self.mid2_ratios,
                self.pass_energies
            ],
            dims=["retardation", "mid1_ratio", "mid2_ratio", "pass_energy"]
        )

        return resolution_xarray

    def create_avg_tof_xarray(self, resolution_dict):
        r_coord = list(range(np.min(self.retardation), np.max(self.retardation) + 1))
        tof_data = np.full(
            (len(r_coord), len(self.mid1_ratios), len(self.mid2_ratios), len(self.pass_energies)),
            np.nan
        )
        retardation_indices = {val: idx for idx, val in enumerate(r_coord)}
        mid1_indices = {val: idx for idx, val in enumerate(self.mid1_ratios)}
        mid2_indices = {val: idx for idx, val in enumerate(self.mid2_ratios)}

        for (retardation, mid1, mid2), gradient in resolution_dict.items():
            ridx = retardation_indices[retardation]
            mid1_idx = mid1_indices[mid1]
            mid2_idx = mid2_indices[mid2]
            tof_data[ridx, mid1_idx, mid2_idx, :] = gradient[0]

        resolution_xarray = xr.DataArray(
            tof_data,
            coords=[
                r_coord,
                self.mid1_ratios,
                self.mid2_ratios,
                self.pass_energies
            ],
            dims=["retardation", "mid1_ratio", "mid2_ratio", "pass_energy"]
        )

        return resolution_xarray

    def create_combined_array(self):
        combined_data = []
        last_length = None

        for entry in self.data:
            initial_ke = entry['initial_ke']
            elevation = entry['initial_elevation']
            retardation = entry['retardation']
            mid1_ratio = entry['mid1_ratio']
            mid2_ratio = entry['mid2_ratio']
            tof = entry['tof_values']
            y_tof = entry['y_tof']
            mask = entry['mask']

            # Ensure all arrays have the same length
            lengths = [
                len(initial_ke),
                len(elevation),
                len(tof),
                len(y_tof),
                len(mask)
            ]

            if not all(length == lengths[0] for length in lengths):
                print(f"Skipping entry due to internal length mismatch: {entry}")
                continue

            # Ensure the length matches the last entry added to combined_data
            current_length = lengths[0]
            if last_length is not None and current_length != last_length:
                print(f"Skipping entry due to length mismatch with previous entry: {entry}")
                continue

            retardation_array = np.full_like(initial_ke, retardation)
            mid1_ratio_array = np.full_like(initial_ke, mid1_ratio)
            mid2_ratio_array = np.full_like(initial_ke, mid2_ratio)
            print(retardation_array.shape, mid2_ratio_array.shape, mid2_ratio_array.shape)

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
            last_length = current_length

        if combined_data:
            combined_array = np.vstack(combined_data)
        else:
            combined_array = np.empty((0, 8))  # Return an empty array if no valid data

        return combined_array


def custom_data_loader(base_dir, x_tof_range=None, y_tof_range=None, retardation_range=None,
                       mid1_range=None, mid2_range=None, overwrite=False):
    all_data = []
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        # Check if it's a directory and matches the pattern 'R' followed by digits
        if os.path.isdir(dir_path) and re.match(r'^R\d+$', dir_name):
            # Process files in this directory
            for file in os.listdir(dir_path):
                if file.endswith('.h5') and file.startswith('sim'):
                    file_path = os.path.join(dir_path, file)
                    retardation, mid1_ratio, mid2_ratio = parse_filename(os.path.basename(file_path))
                    if not retardation:
                        continue
                    if retardation_range and not (retardation_range[0] <= retardation <= retardation_range[1]):
                        continue
                    if mid1_range and not (mid1_range[0] <= mid1_ratio <= mid1_range[1]):
                        continue
                    if mid2_range and not (mid2_range[0] <= mid2_ratio <= mid2_range[1]):
                        continue
                    with h5py.File(file_path, 'r+') as h5f:
                        # Extract data from the HDF5 fil
                        grp = h5f['data1']
                        initial_ke = grp['initial_ke'][:]
                        initial_elevation = grp['initial_elevation'][:]
                        x_tof = grp['x'][:]
                        y_tof = grp['y'][:]
                        tof_values = grp['tof'][:]
                        pass_energy = np.round(initial_ke[0] - np.abs(retardation), decimals=2)

                        # Create initial data entry
                        data_entry = {
                            'initial_ke': initial_ke,
                            'initial_elevation': initial_elevation,
                            'pass_energy': pass_energy,
                            'retardation': retardation,
                            'mid1_ratio': mid1_ratio,
                            'mid2_ratio': mid2_ratio,
                            'tof_values': tof_values,
                            'y_tof': y_tof,
                            'x_tof': x_tof,
                        }
                        if 'collection_efficiency' in grp.attrs and 'ks_score' in grp.attrs and 'mask' in grp and not overwrite:
                            efficiency = grp.attrs['collection_efficiency']
                            ks_score = grp.attrs['ks_score']
                            mask = grp['mask'][:]

                        # Check if x_tof_range and y_tof_range are provided
                        elif x_tof_range is not None:
                            # Create mask
                            mask = create_mask(data_entry, x_tof_range, y_tof_range)
                            grp.create_dataset('mask', data=mask)

                            # Calculate collection efficiency
                            start_len = len(initial_ke)
                            masked_len = np.sum(mask)
                            efficiency = masked_len / start_len if start_len > 0 else 0

                            # Calculate KS score
                            ks_score = calculate_ks_score(data_entry['y_tof'][mask])

                            grp.attrs['collection_efficiency'] = efficiency
                            grp.attrs['ks_score'] = ks_score
                        else:
                            raise ValueError(
                                "At least xtof_range must be provided if mask, KS score, and CE "
                                "not already recorded in the H5 file")
                        data_masked = {
                            key: data_entry[key][mask] if isinstance(data_entry[key], np.ndarray) else data_entry[key]
                            for key in data_entry
                        }
                        data_entry['collection_efficiency'] = efficiency
                        data_masked['collection_efficiency'] = efficiency
                        data_entry['ks_score'] = ks_score
                        data_masked['ks_score'] = ks_score

                        # Add data_entry to all_data
                        data_entry['data_masked'] = data_masked
                        all_data.append(data_entry)
    return all_data

if __name__ == '__main__':
    # Example usage
    # Assuming the files are located in a directory named 'data_files' in the current working directory.
    # Instantiate the DataProcessor with the custom data loader
    data_processor = DataProcessor(data_loader=custom_data_loader)

    # Define ranges
    x_tof_range = (403.6, np.inf)

    # Load data
    base_directory = r"C:\Users\proxi\Documents\coding\TOF_data\TOF_data"
    data_processor.load_data(base_dir=base_directory, x_tof_range=x_tof_range)
    # analysis code
    ex = data_processor.create_efficiency_xarray()
    plot_imagetool(ex.sel({'kinetic_energy': 0.1}))
    save_xarray(ex, r"C:\Users\proxi\Documents\coding\TOF_ML\simulations\TOF_simulation\simion_output\collection_efficiency",
                "collection_efficiency")

