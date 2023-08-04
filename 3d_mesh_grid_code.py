# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:44:55 2023
this code creates a 3D mesh grid plot of Retardation vs. KE vs. Elevation with TOF 
@author: lauren
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.cm as cm

folder_path =r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files\test_files'

file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

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
hit_sensor_data['Retardation'] = hit_sensor_data['Filename'].apply(lambda x: int(re.findall(r'_R(\d+)_', x)[0]))

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))  # Adjust the figure size
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh grid with color based on Time of Flight values (Column 4)
colormap = plt.cm.coolwarm  # Choose the desired colormap
norm = plt.Normalize(hit_sensor_data[hit_sensor_data.columns[4]].min(),
                     hit_sensor_data[hit_sensor_data.columns[4]].max())  # Normalize the values
colors = colormap(norm(hit_sensor_data[hit_sensor_data.columns[4]]))  # Apply the colormap and normalization

ax.plot_trisurf(hit_sensor_data[hit_sensor_data.columns[2]], hit_sensor_data[hit_sensor_data.columns[3]],
                hit_sensor_data[hit_sensor_data.columns[10]], facecolor=colors, shade=False, cmap='coolwarm')

# Set axis labels and title
ax.set_xlabel('Elevation')
ax.set_ylabel('Kinetic Energy')
ax.set_zlabel('Retardation')
ax.set_title('Elevation, Kinetic Energy, and Retardation')

# Create a colorbar legend with 10 tick marks representing all the colors
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, ticks=np.linspace(norm.vmin, norm.vmax, 10),
                    pad=0.2)  # Adjust the pad parameter to move the colorbar to the right
cbar.ax.yaxis.set_tick_params(width=0.5)  # Adjust tick width
cbar.ax.tick_params(axis='both', which='major', labelsize=8)  # Adjust tick label size
cbar.set_label('Time of Flight', rotation=270, labelpad=10)  # Rotate the label and adjust the label pad

# Export the plot as a PDF
output_path = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files\test_files\mesh_grid_plot600.pdf'
plt.savefig(output_path, format='pdf')

# Show the plot
plt.show()
