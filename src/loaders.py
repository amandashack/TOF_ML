"""
Created on Oct 16 2023
this code opens .csv files or h5 files that have already been cleaned up

@author: Amanda
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import re
import copy
from sklearn.preprocessing import KBinsDiscretizer
from scipy.optimize import curve_fit, fsolve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from plotter import get_cmap
import math


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev) ** 2)


def inverse_gaussian(tof_value, amplitude, mean, stddev):
    # The Gaussian function
    gaussian = lambda x: amplitude * np.exp(-((x - mean) / (4 * stddev)) ** 2) - tof_value

    # Numerically solve for the energy
    energy_predicted, = fsolve(gaussian, mean)

    return energy_predicted


def requires_invalid_indxs(func):
    def func_wrapper(*args, **kwargs):
        if getattr(args[0], 'invalid_indxs') is None:
            raise AttributeError("invalid indices are not defined yet.")
        return func(*args, **kwargs)
    return func_wrapper


def interpolator(tofs, pass_ens):
    min_pe = 0
    max_pe = 1200
    x_axis = np.linspace(min_pe, max_pe, num=30000)
    interp_y = []
    for i in range(len(tofs)):
        interp_y.append(np.interp(x_axis, pass_ens[i], tofs[i]))
    return np.array([y for y in interp_y]), x_axis


def train_test_val_loader(fp):
    df = np.array([])
    fp = os.path.abspath(fp)
    for file in os.listdir(fp):
        if file.endswith("h5"):
            path1 = os.path.join(fp, file)
            with h5py.File(path1, 'r') as f:
                df = np.append(df, f['data1']['elevation'][:])
                df = np.append(df, f['data1']['pass'][:])
                df = np.append(df, f['data1']['retardation'][:])
                df = np.append(df, f['data1']['ele*ret'][:])
                df = np.append(df, f['data1']['ele*pass'][:])
                df = np.append(df, f['data1']['pass*ret'][:])
                df = np.append(df, f['data1']['residuals'][:])
                df = np.reshape(df, (-1, len(f['data1']['elevation'][:])))
    return df


class MRCOLoader(object):
    """
    Implements data loading and storage for use plotting
    """

    _TOLERATED_EXTENSIONS = {
        ".h5"
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataframe = None
        self.data_masked = None
        self.spec_dict = {}
        self.spec_masked = {}
        self.mask = {}
        self.tof_values = []
        self.pass_energy = []
        self.initial_ke = []
        self.retardation = []
        self.elevation = []
        self.x_tof = []
        self.y_tof = []
        self.p_bins = []

    def load(self):
        fp = os.path.abspath(self.filepath)
        for file in os.listdir(fp):
            if file.endswith("h5"):
                path1 = os.path.join(fp, file)
                self.retardation.append(int(re.findall(r'_R(\d+)', file)[0]))
                with h5py.File(path1, 'r') as f:
                    x_tof = f['data1']['x'][:]
                    y_tof = f['data1']['y'][:]
                    tof_values = f['data1']['tof'][:]
                    initial_ke = f['data1']['initial_ke'][:]
                    elevation = f['data1']['elevation'][:]
                    pass_energy = list(ke - self.retardation[-1] for ke in initial_ke)
                    self.initial_ke.append(initial_ke.tolist())
                    self.pass_energy.append(pass_energy)
                    self.tof_values.append(tof_values.tolist())
                    self.elevation.append(elevation.tolist())
                    self.x_tof.append(x_tof.tolist())
                    self.y_tof.append(y_tof.tolist())
                    self.spec_dict[self.retardation[-1]] = [self.elevation[-1],
                                                            self.pass_energy[-1],
                                                            self.tof_values[-1]]
            else:
                pass
        self.organize_data()

    def organize_data(self):
        tof_c = copy.deepcopy(self.tof_values)
        pass_c = copy.deepcopy(self.pass_energy)
        ele_c = copy.deepcopy(self.elevation)
        r_c = copy.deepcopy(self.retardation)
        ke_c = copy.deepcopy(self.initial_ke)
        x_tof = copy.deepcopy(self.x_tof)
        y_tof = copy.deepcopy(self.y_tof)
        s = sorted(range(len(r_c)),
                   key=r_c.__getitem__)
        tof_c = [tof_c[i] for i in s]
        pass_c = [pass_c[i] for i in s]
        ele_c = [ele_c[i] for i in s]
        r_c = [r_c[i] for i in s]
        ke_c = [ke_c[i] for i in s]
        x_tof = [x_tof[i] for i in s]
        y_tof = [y_tof[i] for i in s]
        #self.dataframe = self.gen_dataframe(ele_c, pass_c, r_c, tof_c)
        self.spec_dict = self.gen_spec(ele_c, pass_c, r_c, tof_c)
        self.initial_ke = ke_c
        self.retardation = r_c
        self.pass_energy = pass_c
        self.tof_values = tof_c
        self.elevation = ele_c
        self.x_tof = x_tof
        self.y_tof = y_tof

    @staticmethod
    def gen_spec(elevation, pass_energy, retardation, tof):
        spec = {}
        for i in range(len(retardation)):
            spec[retardation[i]] = [elevation[i], pass_energy[i], tof[i]]
        return spec

    @staticmethod
    def gen_dataframe(elevation, pass_energy, retardation, tof):
        pass_flat = []
        tof_flat = []
        r_flat = []
        ele_flat = []
        for i in range(len(retardation)):
            r_flat += len(pass_energy[i]) * [retardation[i]]
            pass_flat += pass_energy[i]
            ele_flat += elevation[i]
            tof_flat += tof[i]
        data = np.array((np.asarray(ele_flat), np.log2(np.asarray(pass_flat)),
                         np.asarray(r_flat), np.log2(np.asarray(tof_flat))))
        return data

    def create_mask(self, x, y, min_pass, mask_name):
        # used to generate a mask for a part of the data
        # x and y are tuples with the first value being the minimum and second the maximum
        m = []
        for i in range(len(self.retardation)):
            xtof = np.asarray(self.x_tof[i])[:].astype(float)
            ytof = np.abs(np.asarray(self.y_tof[i])[:].astype(float))
            pass_en = np.asarray(self.pass_energy[i])[:].astype(float)
            xmin_mask = xtof > x[0]
            xmax_mask = xtof < x[1]
            ymin_mask = ytof > y[0]
            ymax_mask = ytof < y[1]
            pass_mask = pass_en > min_pass
            mask = xmin_mask & xmax_mask & ymin_mask & ymax_mask & pass_mask
            m.append(mask)
        self.mask[mask_name] = m
        self.apply_mask()

    def apply_mask(self):
        tof_c = copy.deepcopy(self.tof_values)
        pass_c = copy.deepcopy(self.pass_energy)
        ele_c = copy.deepcopy(self.elevation)
        for key in self.mask.keys():
            m = self.mask[key]
            for i in range(len(self.retardation)):
                # this does not currently apply multiple masks...
                tof_c[i] = np.asarray(tof_c[i])[m[i]].tolist()
                pass_c[i] = np.asarray(pass_c[i])[m[i]].tolist()
                ele_c[i] = np.asarray(ele_c[i])[m[i]].tolist()
        #self.data_masked = self.gen_dataframe(ele_c, pass_c, self.retardation, tof_c)
        self.spec_masked = self.gen_spec(ele_c, pass_c, self.retardation, tof_c)

    def check_rebalance(self, nbins):
        bs = []
        for i in range(nbins):
            b = self.p_bins == i
            bs.append(np.count_nonzero(b))
        bs.append(sum(bs))
        return bs

    def rebalance(self, ratio, nbins):
        pass_en = []
        for key in self.spec_masked.keys():
            # make list of all pass energy values
            pass_en += self.spec_masked[key][1]
        # convert to log_2 and flatten
        p = np.log2(pass_en).reshape((-1, 1))
        # use KBinsDiscretizer to convert continuous data into intervals
        est = KBinsDiscretizer(n_bins=nbins, encode='ordinal', strategy='uniform',
                               subsample=None)
        est.fit(p)
        # make a discrete mapping
        self.p_bins = est.transform(p)
        # get the length of each mapping
        bs = self.check_rebalance(nbins)
        n_lb = int((1-ratio) * bs[0])
        training = []
        val = []
        test = []
        for i in range(nbins):
            mask = self.p_bins == i
            df_masked = self.data_masked[:, mask.flatten()]
            df_masked_shuffled = shuffle(df_masked.T, random_state=i)
            training.append(df_masked_shuffled[:n_lb, :])
            the_rest = df_masked[:, n_lb:].T
            v, t = train_test_split(the_rest, test_size=0.50, random_state=42, shuffle=True)
            val.append(v)
            test.append(t)
        """
        fig, ax = plt.subplots(1, 1)
        values, bins, bars = plt.hist(training[2][:, 1], edgecolor='white')
        plt.title("histogram of randomly selected values\n from high bin")
        ax.set_xlabel("Discretization")
        ax.set_ylabel("Number of values")
        ax.locator_params(axis='x', integer=True)
        plt.bar_label(bars, fontsize=20, color='navy')
        plt.margins(x=0.01, y=0.1)
        plt.tight_layout()
        plt.show()
        """
        return training, val, test


class DataStructure:
    def __init__(self, filepath):
        """
        Initializes the DataStructure with the path to the directory containing the files.
        :param filepath: Path to the directory containing the HDF5 files.
        """
        self.filepath = filepath
        # Initialize lists to store data
        self.retardation = []
        self.data = []
        self.training_data = []
        self.validation_data = []
        self.test_data = []

    @staticmethod
    def gen_dataframe(retardation, *args):
        """
        Generates a numpy array from given retardation values and corresponding measurements.
        Each argument after retardation is expected to be a list of lists of measurements, where each sublist
        corresponds to measurements for a given retardation value. It's ensured that each list within these
        lists of lists is of the same length across all measurement types before combining them into a 2D numpy array.

        :param retardation: A list of retardation values.
        :param *args: Variable number of lists of lists of measurements corresponding to the retardation values.
        :return: A 2D numpy array where each row corresponds to a different measurement type, and columns
                 aggregate all measurements across retardation values.
        """
        # Ensure that each sublist within the lists of lists has the same length across all args
        sublist_lengths = [len(sublist) for arg in args for sublist in arg][:len(retardation)]

        # Stack the measurement lists horizontally (as columns) for each measurement type
        stacked_measurements = [np.hstack(measurements) for measurements in args]

        # Retardation values expanded for each measurement
        expanded_retardation = np.repeat(retardation, sublist_lengths)

        # Combine the expanded retardation and stacked measurements into a 2D array
        data = np.vstack([expanded_retardation] + stacked_measurements)

        return data

    @staticmethod
    def append_or_create(data_list, r, new_data):
        # Check if a dict with the current retardation already exists
        existing_data = next((d for d in data_list if d['retardation'] == r), None)
        if existing_data is not None:
            # Append the new data to the existing arrays
            existing_data['elevation'] = np.concatenate((existing_data['elevation'], new_data['elevation']))
            existing_data['pass_energy'] = np.concatenate((existing_data['pass_energy'], new_data['pass_energy']))
            existing_data['tof_values'] = np.concatenate((existing_data['tof_values'], new_data['tof_values']))
        else:
            # Create a new dictionary for the retardation and append it to the list
            data_list.append(new_data)

    def load(self, retardation_value=None):
        """
        Loads data from all HDF5 files in the specified directory that match the naming pattern.
        Assumes the HDF5 files have specific datasets within them.
        """
        fp = os.path.abspath(self.filepath)
        for file in os.listdir(fp):
            if file.endswith("h5"):
                path1 = os.path.join(fp, file)
                # Extract retardation value from the file name
                if not retardation_value:
                    try:
                        rv = int(re.findall(r'_R(\d+)', file)[0])
                    except Exception as e:
                        print(e)
                else:
                    rv = retardation_value
                self.retardation.append(rv)
                print(self.retardation)

                with h5py.File(path1, 'r') as f:
                    # Extracting the necessary datasets
                    initial_ke = f['data1']['initial_ke'][:]
                    initial_elevation = f['data1']['initial_elevation'][:]
                    initial_azimuth = f['data1']['initial_azimuth'][:]

                    x_tof = f['data1']['x'][:]
                    y_tof = f['data1']['y'][:]
                    z_tof = f['data1']['z'][:]
                    tof_values = f['data1']['tof'][:]
                    final_elevation = f['data1']['final_elevation'][:]
                    final_azimuth = f['data1']['final_azimuth'][:]
                    final_ke = f['data1']['final_ke'][:]

                    # Calculate the pass energy
                    pass_energy = np.array([ke - self.retardation[-1] for ke in initial_ke])

                    # Append the loaded data in a structured way
                    self.data.append({
                        'initial_ke': initial_ke,
                        'initial_elevation': initial_elevation,
                        'initial_azimuth': initial_azimuth,
                        'pass_energy': pass_energy,
                        'retardation': self.retardation[-1],
                        'x_tof': x_tof,
                        'y_tof': y_tof,
                        'z_tof': z_tof,
                        'tof_values': tof_values,
                        'final_ke': final_ke,
                        'final_elevation': final_elevation,
                        'final_azimuth': final_azimuth,
                    })

    def apply_mask(self, mask):
        """
        Applies a mask to the data based on the given x_tof and y_tof limits.
        :param x_tof_range: A tuple (min, max) defining the limits for x_tof.
        :param y_tof_range: A tuple (min, max) defining the limits for y_tof.
        :param min_pass: A float defining the minimum value for the pass energy.
        """
        for i in range(len(self.data)):
            start_len = len(self.data[i]['initial_ke'])
            for key in self.data[i]:
                if key not in ['retardation']:
                    self.data[i][key] = np.array(self.data[i][key])[mask[i]]
            self.data[i]['collection_efficiency'] = len(self.data[i]['initial_ke'])/start_len

    def create_mask(self, x_tof_range, y_tof_range, min_pass):
        # used to generate a mask for a part of the data
        # x and y are tuples with the first value being the minimum and second the maximum
        m = []
        for i in range(len(self.retardation)):
            xtof = np.asarray(self.data[i]["x_tof"])[:].astype(float)
            ytof = np.abs(np.asarray(self.data[i]["y_tof"])[:].astype(float))
            pass_en = np.asarray(self.data[i]["pass_energy"])[:].astype(float)
            xmin_mask = xtof > x_tof_range[0]
            xmax_mask = xtof < x_tof_range[1]
            ymin_mask = ytof > y_tof_range[0]
            ymax_mask = ytof < y_tof_range[1]
            pass_mask = pass_en > min_pass
            mask = xmin_mask & xmax_mask & ymin_mask & ymax_mask & pass_mask
            m.append(mask)
        self.apply_mask(m)

    def rebalance(self, ratio, nbins, plot=False, verbose=False, plot_training=False):
        """
        Rebalance the data according to the given ratio and number of bins, with optional plotting and printing.
        :param ratio: The percentage of total data to use.
        :param nbins: The number of bins to discretize the log_2(pass_energy).
        :param plot: If True, plots the distribution of pass energy across bins.
        :param verbose: If True, prints information about the data points in each set.
        :param plot_training: If True, plots the training set.
        """
        # Reset the data lists

        df = self.gen_dataframe(self.retardation, [d['elevation'] for d in self.data],
                                [d['pass_energy'] for d in self.data],
                                [d['tof_values'] for d in self.data])
        self.training_data = []
        self.validation_data = []
        self.test_data = []

        pass_en = df[2, :]
        # convert to log_2 and flatten
        p = np.log2(pass_en).reshape((-1, 1))
        # use KBinsDiscretizer to convert continuous data into intervals
        est = KBinsDiscretizer(n_bins=nbins, encode='ordinal', strategy='uniform',
                               subsample=None)
        est.fit(p)
        # make a discrete mapping
        p_bins = est.transform(p)
        # get the length of each mapping
        bin_counts = np.bincount(p_bins.astype(int).flatten(), minlength=nbins)
        n_lb = int((1-ratio) * min(bin_counts))
        for i in range(nbins):
            bin_mask = (p_bins == i)
            df_masked = df[:, bin_mask.flatten()]
            df_masked_shuffled = shuffle(df_masked.T, random_state=i).T
            training = df_masked_shuffled[:, :n_lb]
            the_rest = df_masked[:, n_lb:]
            val, test = train_test_split(the_rest, test_size=0.50, random_state=42, shuffle=True)
            for r in self.retardation:
                r_mask_train = training[0] == r
                r_mask_val = val[:, 0] == r
                r_mask_test = test[:, 0] == r

                # Extract the data for the current retardation and create a dictionary
                data_train = {
                    "retardation": r,
                    "elevation": training[1, r_mask_train],
                    "pass_energy": training[2, r_mask_train],
                    "tof_values": training[3, r_mask_train]
                }
                data_val = {
                    "retardation": r,
                    "elevation": val[:, 1][r_mask_val],
                    "pass_energy": val[:, 2][r_mask_val],
                    "tof_values": val[:, 3][r_mask_val]
                }
                data_test = {
                    "retardation": r,
                    "elevation": test[:, 1][r_mask_test],
                    "pass_energy": test[:, 2][r_mask_test],
                    "tof_values": test[:, 3][r_mask_test]
                }

                # Append or create new data entry for training, validation, and test
                self.append_or_create(self.training_data, r, data_train)
                self.append_or_create(self.validation_data, r, data_val)
                self.append_or_create(self.test_data, r, data_test)

        # Optional plotting of the distribution of pass energy across bins
        if plot:
            fig, ax = plt.subplots()
            bars = ax.bar(range(nbins), bin_counts, width=1.0, edgecolor='black')
            ax.set_title('Distribution of Data Points Across Bins')
            ax.set_xlabel('Bins')
            ax.set_ylabel('Number of Data Points')

            # Add text above each bar
            for bar in bars:
                height = bar.get_height()
                ax.annotate('{}'.format(height),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

            plt.show()

        # Optional plotting of the training data
        if plot_training:
            self.plot_tof_vs_pass_energy(self.training_data, verbose=True)

        if verbose:
            print(f"Total data points: {sum(bin_counts)}")
            print(f"Training set: {len(training[0])} points")
            print(f"Validation set: {len(val[0])} points")
            print(f"Test set: {len(test[0])} points")

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

    def plot_histogram(self, ax, parameter, x_label, title=None, nbins=50):
        cmap = get_cmap(len(self.data))
        for i in range(len(self.data)):
            ax.hist(self.data[i][parameter], bins=nbins, color=cmap(i),
                     edgecolor='black', alpha=0.5, label=f"R={self.data[i]['retardation']}")
        if title:
            ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)

    def plot_radial_distribution(self, nbins=36):
        # Binning the radii
        elevations = np.asarray(self.data[1]['initial_elevation'])
        theta_max = elevations.max()
        print(theta_max)
        r_max = 406.7 * math.tan(theta_max*np.pi/180)
        print(r_max)
        radii = 406.7 * np.tan(elevations*np.pi/180)
        num_bins = nbins
        bin_edges = np.linspace(0, r_max, num_bins + 1)
        bin_counts, _ = np.histogram(radii, bins=bin_edges)

        # Prepare to generate the 2D plot
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)  # Full circle of angles
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

        for i in range(num_bins):
            count = bin_counts[i]
            radius = np.mean([bin_edges[i], bin_edges[i + 1]])  # Average radius for the bin

            # Randomly choose 'count' angles from the angle array
            chosen_angles = np.random.choice(angles, size=count, replace=False)

            # Plot each point at the randomly chosen angle with the same radial distance
            ax.scatter(chosen_angles, np.full_like(chosen_angles, radius), alpha=0.5, s=10)

        ax.set_title('2D Radial Distribution from 1D Slice Spread Across Angles')
        plt.show()

    def plot_resolution(self, ds, specific_retardations=None, verbose=False):

        if specific_retardations is not None:
            data_to_plot = [item for item in ds if item['retardation'] in specific_retardations]
        else:
            data_to_plot = ds

        predicted_energies = []
        fwhm_values = []
        mean_energy_values = []
        for i in range(len(data_to_plot)):
            r = data_to_plot[i]['retardation']
            tof = data_to_plot[i]['tof_values']
            energy = data_to_plot[i]['pass_energy']
            predicted_energies.append({'retardation': r, 'predicted': [], 'actual': []})
            fwhm_values.append({'retardation': r, 'fwhm': []})
            mean_energy_values.append({'retardation': r, 'mean_energy': []})

            # Define the number of bins for TOF data
            n_bins = 600
            bin_edges = np.linspace(tof.min(), tof.max(), n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Fit a Gaussian to the energy distribution within each TOF bin
            fit_params = []
            for j in range(n_bins):
                # Get the indices of the points within the current bin
                bin_indices = np.where((tof >= bin_edges[j]) & (tof < bin_edges[j + 1]))[0]

                # Continue only if there are enough points to fit the Gaussian
                if len(bin_indices) > 3:  # At least 3 points to fit a Gaussian
                    tof_bin = tof[bin_indices]
                    energy_bin = energy[bin_indices]

                    # Use the mean and standard deviation of the energy values as initial fitting parameters
                    initial_guess = [tof_bin.max(), energy_bin.mean(), energy_bin.std()]

                    try:
                        # Fit a Gaussian to the TOF values
                        popt, _ = curve_fit(gaussian, energy_bin, tof_bin, p0=initial_guess, maxfev=800)
                        fit_params.append(popt)
                        fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(popt[2])
                        fwhm_values[i]["fwhm"] += [fwhm]
                        mean_energy_values[i]['mean_energy'] += [energy_bin.mean()]

                        # Generate x values for the fitted curve
                        energy_fit = np.linspace(energy_bin.min(), energy_bin.max(), 100)
                        tof_fit = gaussian(energy_fit, *popt)


                        # Plot the fitted curve
                        plt.plot(energy_fit, tof_fit)
                    except RuntimeError as e:
                        fit_params.append([np.nan, np.nan, np.nan])
                        # Skip if the fit did not converge (indicated by NaNs)
                    if np.isnan(fit_params[-1]).any():
                        continue
                    predicted_energy = []
                    for tof_value in tof_bin:
                        amplitude, mean, stddev = fit_params[-1]
                        predicted_energy.append(inverse_gaussian(tof_value, amplitude, mean, stddev))
                    predicted_energies[i]['predicted'] += predicted_energy
                    predicted_energies[i]['actual'] += energy_bin.flatten().tolist()
                else:
                    energy_bin = energy[bin_indices]
                    predicted_energies[i]['predicted'] += ([np.nan]*len(bin_indices))
                    predicted_energies[i]['actual'] += energy_bin.flatten().tolist()

        for item in data_to_plot:
            plt.scatter(item['pass_energy'], item['tof_values'], alpha=0.6,
                        label=f"R={item['retardation']}")
        plt.xlabel('TOF')
        plt.ylabel('Energy')
        plt.title('Gaussian Fits to Binned TOF vs. Energy Data')
        plt.legend()
        plt.show()

        # Now plot FWHM vs Mean Energy for each bin
        plt.figure()
        for i in range(len(mean_energy_values)):
            plt.scatter(mean_energy_values[i]['mean_energy'], fwhm_values[i]['fwhm'],
                        label=f"R={fwhm_values[i]['retardation']}")
        plt.title('FWHM vs Mean Energy')
        plt.xlabel('Mean Energy (eV)')
        plt.ylabel('FWHM')
        plt.grid(True)
        plt.show()

        plt.figure()
        for item in predicted_energies:
            plt.scatter(item['actual'], item['predicted'], alpha=0.6, label=f"R={item['retardation']}")
        plt.title('Actual vs Predicted Energy')
        plt.xlabel('Actual Energy (eV)')
        plt.ylabel('Predicted Energy (eV)')
        plt.grid(True)
        plt.show()

    def get_data(self):
        """
        Returns the loaded data.
        """
        return self.data

def plot_relation2(ax, ds, x_label, y_label, title=None, plot_log=False):
    """
    # I think I made this for looking at old data
    """
    for item in ds:
        if not plot_log:
            ax.scatter(ds[item][1], ds[item][2], alpha=0.6, label=f"R={item}")
        else:
            ax.scatter(np.log2(ds[item][1]), np.log2(ds[item][2]), alpha=0.5, label=f"R={item}")
    if title:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True)

if __name__ == '__main__':
    # Example usage
    # Assuming the files are located in a directory named 'data_files' in the current working directory.
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #amanda_filepath = dir_path + "\\NM_simulations"
    dir_path = "C:/Users/proxi/Documents/coding/TOF_ML/simulations/TOF_simulation"
    #dir_path = "C:/Users/proxi/Documents/coding/TOF_ML/src/NM_simulations"
    amanda_filepath = dir_path + "/simion_output/positive_voltage"
    data_loader = DataStructure(filepath=amanda_filepath)
    data_loader.load()
    #fig, ax = plt.subplots()
    #plot_relation2(ax, data_loader.spec_masked, "log(Pass Energy)","log(Time of Flight)", plot_log=True)
    #plt.show()
    #data_loader.plot_histogram(ax, "initial_ke", "Kinetic Energy (eV)")
    #plt.show()
    #print(data_loader.data[0]['initial_ke'].tolist(), '\n\n', data_loader.data[0]['pass_energy'].tolist())
    #data_loader.plot_tof_vs_pass_energy(data_loader.data, verbose=True)
    data_loader.create_mask(x_tof_range=(403.5, np.inf), y_tof_range=(-13.5, 13.5), min_pass=0)
    #data_loader.rebalance(ratio=0.2, nbins=3, plot=False, verbose=False, plot_training=False)
    #data_loader.plot_tof_vs_pass_energy(data_loader.data, verbose=True)
    #data_loader.plot_resolution(data_loader.data)


    # Example of using the function

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Create a figure with multiple subplots
    ax = axes.flatten()
    data_loader.plot_histogram(ax[0], "pass_energy", "Pass Energy (eV)", title=None, nbins=50)
    data_loader.plot_histogram(ax[1], "x_tof", "x position", title=None, nbins=50)
    data_loader.plot_histogram(ax[2], "y_tof", "y position", title=None, nbins=50)
    data_loader.plot_histogram(ax[3], "initial_elevation", "initial elevation", title=None, nbins=50)
    plt.show()

    fig, ax = plt.subplots()
    data_loader.plot_relation(ax, data_loader.data, "pass_energy", "tof_values", "log(Pass Energy)",
                              "log(Time of Flight)", plot_log=True)
    plt.show()
    data_loader.plot_radial_distribution()


