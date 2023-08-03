# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:44:24 2023

@author: lauren
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Setting up hit_sensor_data
folder_path = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files'

file_list = [file for file in os.listdir(folder_path) if file.endswith('grouped.csv')]

dfs = []

for file in file_list:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df['Filename'] = file  # Add Filename column to the DataFrame
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

combined_df[combined_df.columns[5]] = pd.to_numeric(combined_df[combined_df.columns[5]], errors='coerce')
combined_df[combined_df.columns[6]] = pd.to_numeric(combined_df[combined_df.columns[6]], errors='coerce')

# Convert 'Time of Flight' column to numeric type
combined_df[combined_df.columns[4]] = pd.to_numeric(combined_df[combined_df.columns[4]], errors='coerce')

hit_sensor_data = combined_df[
    (combined_df[combined_df.columns[5]].notnull()) &
    (combined_df[combined_df.columns[5]].astype(float) >= 402) &
    (combined_df[combined_df.columns[6]].notnull()) &
    (combined_df[combined_df.columns[6]].astype(float).between(-13.7, 13.7))
].copy()

# Extract the retardation value from each file name and convert to numeric
hit_sensor_data['Retardation'] = hit_sensor_data['Filename'].apply(lambda x: int(re.findall(r'_R(\d+)_', x)[0]) if re.findall(r'_R(\d+)_', x) else None)

# Extract the data from the DataFrame without rounding
x_data = pd.to_numeric(hit_sensor_data.iloc[:, 2], errors='coerce')
y_data = pd.to_numeric(hit_sensor_data.iloc[:, 3], errors='coerce')
z_data = pd.to_numeric(hit_sensor_data.iloc[:, 10], errors='coerce')
tof_data = pd.to_numeric(hit_sensor_data.iloc[:, 4], errors='coerce')

# Remove NaN values
x_data = x_data.dropna()
y_data = y_data.dropna()
z_data = z_data.dropna()
tof_data = tof_data.dropna()

# Calculate the bin edges for the y-axis
#y_bins = np.linspace(min(y_data), max(y_data), 20)
y_bins = np.floor(np.linspace(min(y_data), max(y_data), 20))

# Create colormap for TOF values
cmap = plt.cm.get_cmap('gist_rainbow')
norm = Normalize(vmin=min(tof_data), vmax=max(tof_data))
colors = cmap(norm(tof_data))

# Plot the 3D bars with colors if hit_sensor_data is not empty
if not hit_sensor_data.empty and not x_data.empty:
    # Plot the 3D bars with colors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x_data, y_data, np.zeros_like(z_data), 1, 1, z_data, color=colors)

    # Set labels for the axes
    ax.set_xlabel('Elevation')
    ax.set_ylabel('Kinetic Energy')
    ax.set_zlabel('Retardation')
    ax.set_title('Elevation, Kinetic Energy, and Retardation')

    # Create a colorbar legend
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=np.linspace(norm.vmin, norm.vmax, 10), pad=0.2)
    cbar.ax.yaxis.set_tick_params(width=0.5)
    cbar.ax.tick_params(axis='both', which='major', labelsize=8)
    cbar.set_label('Time of Flight', rotation=270, labelpad=10)

    # Export the plot as a PDF
    output_path = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files\histo_plot_hi.pdf'
    plt.savefig(output_path, format='pdf')

# Show the plot
plt.show()

