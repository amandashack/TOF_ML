import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic, norm
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from loaders import DataStructure


def plot_statistics(data, bins):
    # Binning the data based on y into 10 quantile bins
    bin_counts, bin_edges = np.histogram(data['tof_values'], bins=bins)
    bin_indices = np.digitize(data['tof_values'], bin_edges[:-1])

    # Compute means and standard deviations in each bin for plotting
    means = []
    stddevs = []
    binned_data = [[] for _ in range(len(bins))]
    for x_val, bin_idx in zip(data['pass_energy'], bin_indices):
        if bin_idx > 0:  # to avoid issues with binning on the edge
            binned_data[bin_idx-1].append(x_val)
    # Fitting data in the selected middle bin for Gaussian and generating plot data
    selected_bin_data = binned_data[10]
    mu, std = norm.fit(selected_bin_data)
    x_range = np.linspace(min(selected_bin_data), max(selected_bin_data), 100)
    gauss_fit = norm.pdf(x_range, mu, std)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot of binned data
    colors = plt.cm.viridis(np.linspace(0, 1, len(bins)))
    for i, (bin_data, color) in enumerate(zip(binned_data, colors), 1):
        ax1.scatter(bin_data, [bin_edges[i-1]]*len(bin_data), color=color, alpha=0.6)

    ax1.set_xlabel('Pass Energy (eV)')
    ax1.set_ylabel('Time of Flight')
    ax1.set_title('Scatter plot of TOF vs Pass with Binned TOF')
    ax1.grid(True)

    # Histogram and Gaussian fit for the selected bin
    ax2.hist(selected_bin_data, bins='auto', density=True, color='g', alpha=0.6, label='Data histogram')
    ax2.plot(x_range, gauss_fit, 'k', linewidth=2, label=f'Fit: $\\mu={mu:.2f}$, $\\sigma={std:.2f}$')
    ax2.set_title('Fit results for middle bin')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Density')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def calculate_nbins(tof_values, time_resolution):
    tof_values = sorted(tof_values)
    width = tof_values[-1] - tof_values[0]
    resolution = int(width/time_resolution)
    return np.linspace(tof_values[0], tof_values[-1], resolution)

dir_path = "C:/Users/proxi/Documents/coding/simulations/TOF_simulation"
amanda_filepath = dir_path + "/simion_output"
data_loader = DataStructure(filepath=amanda_filepath)
data_loader.load()
data_loader.create_mask(x_tof_range=(403.5, np.inf), y_tof_range=(-13.5, 13.5), min_pass=0)
bins = calculate_nbins(data_loader.data[1]['tof_values'].tolist(), 0.005)
plot_statistics(data_loader.data[2], bins)
