from ..loaders import DataStructure
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic, norm


def plot_statistics(data, nbins=50):
    # Binning the data based on y into 10 quantile bins
    bin_counts, bin_edges = np.histogram(data['tof_values'], bins=nbins)
    bin_indices = np.digitize(data['tof_values'], bin_edges[:-1])

    # Compute means and standard deviations in each bin for plotting
    means = []
    stddevs = []
    binned_data = [[] for _ in range(nbins)]
    for x_val, bin_idx in zip(data['pass_energy'], bin_indices):
        if bin_idx > 0:  # to avoid issues with binning on the edge
            binned_data[bin_idx-1].append(x_val)

    # Fitting data in the selected middle bin for Gaussian and generating plot data
    selected_bin_data = binned_data[int(nbins/2)]  # Middle bin approx
    mu, std = norm.fit(selected_bin_data)
    x_range = np.linspace(min(selected_bin_data), max(selected_bin_data), 100)
    gauss_fit = norm.pdf(x_range, mu, std)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot of binned data
    colors = plt.cm.viridis(np.linspace(0, 1, nbins))
    for i, (bin_data, color) in enumerate(zip(binned_data, colors), 1):
        ax1.scatter([bin_edges[i-1]]*len(bin_data), bin_data, color=color, label=f'Bin {i}', alpha=0.6)

    ax1.set_xlabel('Pass Energy (eV)')
    ax1.set_ylabel('Time of Flight')
    ax1.set_title('Scatter plot of TOF vs Pass with Binned TOF')
    ax1.legend(title="Y Bins")
    ax1.grid(True)

    # Histogram and Gaussian fit for the selected bin
    ax2.hist(selected_bin_data, bins=15, density=True, color='g', alpha=0.6, label='Data histogram')
    ax2.plot(x_range, gauss_fit, 'k', linewidth=2, label=f'Fit: $\\mu={mu:.2f}$, $\\sigma={std:.2f}$')
    ax2.set_title('Fit results for middle bin')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Density')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def calculate_nbins(tof_values):
    time_resolution = .000000001  # 1 nanosecond
    tof_values = sorted(tof_values)
    width = tof_values[-1] - tof_values[0]
    return int(width/time_resolution)

dir_path = "C:/Users/proxi/Documents/coding/TOF_ML/simulations/TOF_simulation"
amanda_filepath = dir_path + "/simion_output"
data_loader = DataStructure(filepath=amanda_filepath)
data_loader.load()
print(data_loader.data[1]['tof_values'])
nbins = calculate_nbins(data_loader.data[1]['tof_values'].tolist())
print(nbins)
plot_statistics(data_loader.data[0], 50)
