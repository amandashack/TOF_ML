import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import sys
sys.path.append('/sdf/home/a/ajshack/TOF_ML/src/utilities')
from plotting_tools import plot_pass_energy_vs_tof

# Specify the HDF5 file path
data_file = '/sdf/scratch/users/a/ajshack/combined_data_large.h5'

# Open the HDF5 file and read the data
with h5py.File(data_file, 'r') as hf:
    # List available datasets in the HDF5 file
    print("Available datasets:", list(hf.keys()))
    # Assuming the dataset is named 'combined_data'
    data = hf['combined_data'][:]

# Extract the necessary columns
# Column indices:
# 0: pass energy (initial_energy)
# 1: elevation
# 2: retardation
# 3: mid1 voltage
# 4: mid2 voltage
# 5: time of flight
# 6: mask

initial_energy = data[:, 0]
tof = data[:, 5]
retardation = data[:, 2]  # Retain the sign of retardation
mask = data[:, -1].astype(bool)  # Convert mask to boolean


# Compute pass_energy based on the sign of retardation
pass_energy = np.where(
    retardation > 0,
    np.round(initial_energy, 2),                 # For positive retardation
    np.round(initial_energy + retardation, 2)    # For negative retardation
)

# Unique values (as provided)
unique_retardation = np.array([
    -2000, -1500, -1000, -856, -676, -526, -375, -250, -180, -120,
    -80, -40, -20, -10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 5, 10, 15
])
unique_pass_energy = np.array([0.3, 0.6, 1., 2., 4., 6., 8., 12., 16., 25., 50., 75., 100., 150., 300.])

print("Unique Retardations:", unique_retardation)
print("Unique Pass Energies:", unique_pass_energy)

# Create DataFrames for data kept and data masked out
# Data kept after masking (mask == True)
df_kept = pd.DataFrame({
    'pass_energy': pass_energy[mask],
    'tof': tof[mask],
    'retardation': retardation[mask]
})

# Data masked out (mask == False)
df_masked_out = pd.DataFrame({
    'pass_energy': pass_energy[~mask],
    'tof': tof[~mask],
    'retardation': retardation[~mask]
})

plot_pass_energy_vs_tof(df_kept, sample_size = 1000)
