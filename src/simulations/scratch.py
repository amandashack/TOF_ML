import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import sys
import json
import xarray as xr
from load_positive_voltages import DS_positive
sys.path.insert(0, os.path.abspath('..'))
from loaders.load_and_save import save_xarray, load_xarray
from utilities.plotting_tools import (plot_imagetool, plot_relation, plot_heatmap, plot_histogram,
                                      plot_energy_resolution, plot_parallel_coordinates, plot_ks_score)
from utilities.mask_data import create_mask
from utilities.calculation_tools import calculate_ks_score, normalize_3D


# decide what you want to load in
#location = r"C:\Users\proxi\Documents\coding\TOF_ML\simulations\TOF_simulation\simion_output\collection_efficiency"
#efficiency_xarray = load_xarray(location, "collection_efficiency")

data_loader = DS_positive()
data_loader.load_data('simulation_data.json', xtof_range=(403.6, np.inf), ytof_range=(-13.74, 13.74),
                      retardation_range=(-10, 10), mid1_range=(0.11248, 0.11248), mid2_range=(0.1354, 0.1354),
                      overwrite=False)
location = r"C:\Users\proxi\Documents\coding\TOF_ML\figures\shack"
#avg_ks_scores = plot_ks_score(data_loader.data_masked, bootstrap=10, directory=location, filename="NM_ks_scores.pdf")

# decide what you want to plot
filtered_data = [entry for entry in data_loader.data_masked if entry['retardation'] == 1
                 and entry['kinetic_energy'] == 2]
fig, ax = plt.subplots()
ax.hist(filtered_data[0]['tof_values'], bins=50, edgecolor='black', alpha=0.5)
plt.show()
