# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:35:43 2023

@author: lauren
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np

folder_path = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files'

file_list = [file for file in os.listdir(folder_path) if file.endswith('grouped.csv')]

dfs = []

for file in file_list:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df['Filename'] = file  # Add Filename column to the DataFrame
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

combined_df[df.columns[5]] = pd.to_numeric(combined_df[df.columns[5]], errors='coerce')
combined_df[df.columns[6]] = pd.to_numeric(combined_df[df.columns[6]], errors='coerce')

# Convert 'Time of Flight' column to numeric type
combined_df[df.columns[4]] = pd.to_numeric(combined_df[df.columns[4]], errors='coerce')

hit_sensor_data = combined_df[
    (combined_df[df.columns[5]].notnull()) &
    (combined_df[df.columns[5]].astype(float) >= 402) &
    (combined_df[df.columns[6]].notnull()) &
    (combined_df[df.columns[6]].astype(float).between(-13.7, 13.7))
].copy()

# Extract the retardation value from each file name and convert to numeric
hit_sensor_data['Retardation'] = hit_sensor_data['Filename'].apply(lambda x: int(re.findall(r'_R(\d+)_', x)[0]))

# Get unique retardation values and assign colors
retardation_values = sorted(hit_sensor_data['Retardation'].unique())
color_range = np.linspace(0, 1, len(retardation_values))
colors = plt.cm.autumn(color_range)

# Create a color dictionary to map retardation values to colors
color_dict = {value: color for value, color in zip(retardation_values, colors)}

# Calculate the reduced kinetic energy by subtracting retardation from the 4th column value
hit_sensor_data['Reduced_Kinetic_Energy'] = hit_sensor_data.iloc[:, 3].astype(float) - hit_sensor_data['Retardation']

# Get unique values in column three and sort them in ascending order
column_three_values = sorted(hit_sensor_data[df.columns[2]].unique())

plots_per_page = 3
total_plots = len(column_three_values)
total_pages = (total_plots - 1) // plots_per_page + 1

# Find the minimum and maximum values of the fifth column for consistent y-axis limits
y_min = hit_sensor_data[df.columns[4]].min()
y_max = hit_sensor_data[df.columns[4]].max()

# Calculate the overall minimum and maximum values of Reduced Kinetic Energy for consistent x-axis limits
x_min = hit_sensor_data['Reduced_Kinetic_Energy'].min()
x_max = hit_sensor_data['Reduced_Kinetic_Energy'].max()

pdf_path = os.path.join(folder_path, 'plot91.pdf')
with PdfPages(pdf_path) as pdf:
    # Variables to store the overall y-axis limits and tick values
    overall_y_min = float('inf')
    overall_y_max = float('-inf')
    y_ticks = None

    for page in range(total_pages):
        start_idx = page * plots_per_page
        end_idx = min((page + 1) * plots_per_page, total_plots)

        fig, axes = plt.subplots(1, plots_per_page, figsize=(12, 8))

        for idx, value in enumerate(column_three_values[start_idx:end_idx], start=1):
            plot_index = start_idx + idx
            ax = axes[idx - 1]

            # Filter the data for the specific column three value
            filtered_data = hit_sensor_data[hit_sensor_data[df.columns[2]] == value]

            # Extract the reduced kinetic energy and fifth column values
            reduced_kinetic_energy = filtered_data['Reduced_Kinetic_Energy']
            fifth_column_values = filtered_data[df.columns[4]]
            retardation_values = filtered_data['Retardation']

            # Assign colors based on retardation values
            colors = [color_dict[retardation] for retardation in retardation_values]

            # Create scatter plot with specific colors
            ax.scatter(reduced_kinetic_energy, fifth_column_values, c=colors)

            ax.set_xlabel('Pass Energy')
            ax.set_ylabel('Time of Flight')
            ax.set_title(f'Elevation: {value}')

            # Update overall y-axis limits
            overall_y_min = min(overall_y_min, fifth_column_values.min())
            overall_y_max = max(overall_y_max, fifth_column_values.max())

        plt.tight_layout()

        # Set the y-axis limits based on the overall min and max values
        overall_y_range = overall_y_max - overall_y_min
        overall_y_min -= overall_y_range * 0.05  # Add a small margin
        overall_y_max += overall_y_range * 0.05  # Add a small margin

        for ax in axes:
            # Set the y-axis limits
            ax.set_ylim(overall_y_min, overall_y_max)

            if y_ticks is None:
                # Store the y-axis tick values from the first plot
                y_ticks = ax.get_yticks()

            # Set the y-axis tick values for consistent scale
            ax.set_yticks(y_ticks[:20])

            # Set the x-axis limits and tick values
            ax.set_xlim(x_min, x_max)

        # Create a legend for the retardation values and their colors
        legend_elements = []
        for value, color in color_dict.items():
            legend_elements.append(Patch(facecolor=color, label=f'Retardation: {value}'))

        plt.legend(handles=legend_elements)

        pdf.savefig(fig)
        plt.close()
        
        plt.show()