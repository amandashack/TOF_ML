from loaders import MRCOLoader, train_test_val_loader
import os
import numpy as np
from model_gen import y0_NM
import h5py
import plotter
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def generate_files(x_data, y_data, output_dir, filename):
    output_file_path = os.path.join(output_dir, filename)
    with h5py.File(output_file_path, "w") as f:
        g1 = f.create_group("data1")
        g1.create_dataset("elevation", data=x_data[:, 0].flatten())
        g1.create_dataset("pass", data=x_data[:, 1].flatten())
        g1.create_dataset("retardation", data=x_data[:, 2].flatten())
        g1.create_dataset("ele*ret", data=(x_data[:, 0].flatten() * x_data[:, 2].flatten()))
        g1.create_dataset("ele*pass", data=(x_data[:, 0].flatten() * x_data[:, 1].flatten()))
        g1.create_dataset("pass*ret", data=(x_data[:, 1].flatten() * x_data[:, 2].flatten()))
        g1.create_dataset("residuals", data=y_data[:])
        print("Data exported to:", output_file_path)


def check_rebalance(data):
    hb = data>7.59
    high_bin= np.count_nonzero(hb)
    mb = np.logical_and(data>=4.96, data<7.59)
    mid_bin = np.count_nonzero(mb)
    lb = data<4.96
    low_bin = np.count_nonzero(lb)
    print(high_bin, mid_bin, low_bin, high_bin + mid_bin + low_bin)


def run_datasplit():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    amanda_filepath = dir_path + "\\NM_simulations"
    print(amanda_filepath)
    multi_retardation_sim = MRCOLoader(amanda_filepath)
    multi_retardation_sim.load()
    multi_retardation_sim.create_mask((402, np.inf), (0, 17.7), 5, "make it")
    training, val, test = multi_retardation_sim.rebalance(0.25, 3)
    x_train = training[0][:, 0:3]
    y_train = training[0][:, 3].T - y0_NM(training[0][:, 1].T)
    x_val = val[0][:, 0:3]
    y_val = val[0][:, 3].T - y0_NM(val[0][:, 1].T)
    x_test = test[0][:, 0:3]
    y_test = test[0][:, 3].T - y0_NM(test[0][:, 1].T)
    for i in range(1, 3):
        x_train = np.append(x_train, training[i][:, 0:3], axis=0)
        y_train = np.append(y_train, training[i][:, 3].T - y0_NM(training[i][:, 1].T), axis=0)
        x_val = np.append(x_val, val[i][:, 0:3], axis=0)
        y_val = np.append(y_val, val[i][:, 3].T - y0_NM(val[i][:, 1].T), axis=0)
        x_test = np.append(x_test, test[i][:, 0:3], axis=0)
        y_test = np.append(y_test, test[i][:, 3].T - y0_NM(test[i][:, 1].T), axis=0)
    train_dir_path = dir_path + "\\NM_simulations\\masked_data3\\train"
    test_dir_path = dir_path + "\\NM_simulations\\masked_data3\\test"
    validate_dir_path = dir_path + "\\NM_simulations\\masked_data3\\validate"
    generate_files(x_train, y_train, train_dir_path, "train_data.h5")
    generate_files(x_test, y_test, test_dir_path, "test_data.h5")
    generate_files(x_val, y_val, validate_dir_path, "validate_data.h5")


def multi_scatter(df):
    fig, ax = plt.subplots()
    ax.scatter(df[0], df[1])
    fig.tight_layout()
    plt.legend(prop={'size': 6})
    plt.show()


def plot_rebalanced_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    train_filepath = dir_path + "/NM_simulations/masked_data3/train"
    train_data = train_test_val_loader(train_filepath)
    fig, ax = plt.subplots()
    df = pd.DataFrame(data=train_data[[1,2,-1], :].T, columns=['pass', 'retardation', 'residuals'])
    vals = df.retardation.unique()
    for val in vals:
        plot_df = pd.DataFrame(df.loc[df['retardation'] == 0])
    plotter.plot_residuals(ax, df, "Train Data", '$log_{2}$(Pass Energy)', "Residuals")
    fig.tight_layout()
    plt.legend(prop={'size': 6})
    plt.show()
    #multi_scatter([X, Y])


def plot_energy_resolutions(resolution):
    plt.figure(figsize=(10, 6))  # Set the figure size

    for idx, (pes, res) in enumerate(resolution):
        # Each pes, res pair corresponds to a different retardation
        # 'idx' can be used to differentiate them in the plot
        plt.scatter(pes, res, label=f'Retardation {idx + 1}')

    plt.xlabel('Pass Energy (eV)')  # X-axis label
    plt.ylabel('Energy Resolution (ΔE)')  # Y-axis label
    plt.title('Energy Resolution vs. Pass Energy for Different Retardations')  # Plot title
    plt.legend()  # Show legend to differentiate between retardations
    plt.grid(True)  # Add a grid for better readability
    plt.show()  # Display the plot


def calculate_delta_E(tof_e1, tof_e2, delta_t, delta_e):
    """
    Calculate ΔE using the given formula:
    ΔE = Δt [δE / (TOF(E) - TOF(E + δE))]

    Parameters:
    tof_e1 (float): Time of flight at energy E
    tof_e2 (float): Time of flight at energy E + delta_e
    delta_t (float): spread in time of flight for energy E
    delta_e (float): energy step between E1 and E2

    Returns:
    float: Calculated energy resolution ΔE
    """
    # TOF(E) is the time of flight at energy E (tof_e)

    delta_E = delta_t * (delta_e / (tof_e1 - tof_e2))
    return delta_E


def plot_energy_resolution():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    amanda_filepath = dir_path + "\\NM_simulations"
    multi_retardation_sim = MRCOLoader(amanda_filepath)
    multi_retardation_sim.load()
    multi_retardation_sim.create_mask((402, np.inf), (0, 17.7), 5, "make it")
    # get the dictionary based on the retardations
    spec_dict = multi_retardation_sim.spec_masked
    # generate the energy resolution for each retardation
    resolution = []
    for r in spec_dict.keys():
        # Find the unique energies and their indices
        initial_pe = np.array(spec_dict[r][1])
        tof_values = np.array(spec_dict[r][2])
        unique_pe, indices, counts = np.unique(initial_pe, return_inverse=True, return_counts=True)

        # Filtering energies based on occurrence
        unique_pe_filtered = unique_pe[counts > 1]
        # Getting indices for filtered energies
        indices_filtered = np.where(np.isin(initial_pe, unique_pe_filtered))[0]

        avg_tof = []
        spread_tof = []
        for pe in unique_pe_filtered:
            idxs = np.where(initial_pe == pe)[0]
            avg_tof.append(np.mean(tof_values[idxs]))
            spread_tof.append(np.ptp(tof_values[idxs]))

        # Get indices that would sort unique_pe_filtered
        sort_indices = np.argsort(unique_pe_filtered)
        # Use these indices to sort both unique_pe_filtered and spread_tof_values_filtered
        sorted_unique_pe_filtered = np.array(unique_pe_filtered)[sort_indices].tolist()
        sorted_spread_tof = np.array(spread_tof)[sort_indices].tolist()
        # Sorting the avg_tof alongside the sorted_unique_pe_filtered and sorted_spread_tof_values_filtered
        sorted_avg_tof = np.array(avg_tof)[sort_indices].tolist()
        # Calculate delta_e for each pair of adjacent energies in the sorted_unique_pe_filtered list
        delta_e_list = np.diff(sorted_unique_pe_filtered)
        # Calculate ΔE for each pair of adjacent energies using the corresponding delta_e
        dE = []
        filtered_pes = []
        for i in range(len(delta_e_list)):
            delta_E = calculate_delta_E(sorted_avg_tof[i], sorted_avg_tof[i + 1], sorted_spread_tof[i], delta_e_list[i])
            if 0.01 <= delta_E <= 5:
                dE.append(delta_E)
                filtered_pes.append(sorted_unique_pe_filtered[i])

        # Ensure that the last pass energy is also included if its delta_E is within the specified range
        last_delta_E = calculate_delta_E(sorted_avg_tof[-1], sorted_avg_tof[-1] + delta_e_list[-1],
                                         sorted_spread_tof[-1], delta_e_list[-1])
        if 0.01 <= last_delta_E <= 5:
            dE.append(last_delta_E)
            filtered_pes.append(sorted_unique_pe_filtered[-1])

        resolution.append([filtered_pes[:100], dE[:100]])
    plot_energy_resolutions(resolution)


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# Function to fit Gaussians to y_data vs. x_data within histogram bins
def fit_gaussian_to_bins(y_data, x_data, bins='auto'):
    # Generating histogram to determine bin edges
    counts, bin_edges = np.histogram(y_data, bins=bins)

    # Container for fit results
    fit_params = []

    # Fit a Gaussian for each bin
    for i in range(len(bin_edges) - 1):
        # Identifying points within the current bin
        bin_mask = (y_data >= bin_edges[i]) & (y_data < bin_edges[i + 1])
        x_in_bin = x_data[bin_mask]
        y_in_bin = y_data[bin_mask]

        if len(x_in_bin) < 3:  # Need at least 3 points to fit a model
            fit_params.append(None)
            continue

        # Initial guesses: A = max count, mu = mean of x_in_bin, sigma = std of x_in_bin
        initial_guess = [max(y_in_bin), np.mean(x_in_bin), np.std(x_in_bin)]

        try:
            # Fitting the Gaussian model
            popt, _ = curve_fit(gaussian, x_in_bin, y_in_bin, p0=initial_guess)
            fit_params.append([x_in_bin, y_in_bin, popt])
        except RuntimeError:
            # In case the fit fails
            print(f"Fit failed for bin {i}")
            fit_params.append(None)

    return fit_params, bin_edges

def plot_energy_v_splat():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    amanda_filepath = dir_path + "\\NM_simulations"
    multi_retardation_sim = MRCOLoader(amanda_filepath)
    multi_retardation_sim.load()
    multi_retardation_sim.create_mask((402, np.inf), (0, 17.7), 5, "make it")
    # get the dictionary based on the retardations
    spec_dict = multi_retardation_sim.spec_masked
    r = list(spec_dict.keys())
    initial_pe = np.array(spec_dict[r[3]][1])
    tof_values = np.array(spec_dict[r[3]][2])

    fit_results, bin_edges = fit_gaussian_to_bins(tof_values, initial_pe, bins='auto')
    plt.figure(figsize=(12, 8))
    # Plotting Gaussian fits for each bin
    print(len(fit_results))
    for i, result in enumerate(fit_results[100:105]):
        if result is not None:
            # Extracting the Gaussian parameters
            A, mu, sigma = result[2]
            # Generating Gaussian curve for the current bin
            y_gaussian = gaussian(result[0], A, mu, sigma)
            plt.scatter(result[0], result[1], alpha=0.5, label='Original Data')
            plt.plot(result[0], y_gaussian, label=f'Bin {i + 1} Fit, sigma = {sigma}')

    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.title('Gaussian Fits to Data in Each Bin')
    plt.legend()
    plt.show()

def plot_raw_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    amanda_filepath = dir_path + "\\NM_simulations"
    multi_retardation_sim = MRCOLoader(amanda_filepath)
    multi_retardation_sim.load()

if __name__ == '__main__':
    plot_raw_data()