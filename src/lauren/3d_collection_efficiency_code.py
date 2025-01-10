import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize

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
combined_df[combined_df.columns[3]] = pd.to_numeric(combined_df[combined_df.columns[3]], errors='coerce')

hit_sensor_data = combined_df[
    (combined_df[combined_df.columns[5]].notnull()) &
    (combined_df[combined_df.columns[5]].astype(float) >= 402) &
    (combined_df[combined_df.columns[6]].notnull()) &
    (combined_df[combined_df.columns[6]].astype(float).between(-13.7, 13.7))
].copy()

# Extract the retardation value from each file name and convert to numeric
retardation_pattern = r'_R(\d+)_'
hit_sensor_data['Retardation'] = hit_sensor_data['Filename'].str.extract(retardation_pattern).astype(float)

# Convert pass energy and time of flight columns to numeric type
hit_sensor_data[hit_sensor_data.columns[3]] = pd.to_numeric(hit_sensor_data[hit_sensor_data.columns[3]], errors='coerce')
hit_sensor_data[hit_sensor_data.columns[10]] = pd.to_numeric(hit_sensor_data[hit_sensor_data.columns[10]], errors='coerce')

# Calculate pass energy and time of flight
hit_sensor_data['Pass Energy'] = hit_sensor_data.iloc[:, 3] - hit_sensor_data.iloc[:, 10]
hit_sensor_data['Time of Flight'] = hit_sensor_data.iloc[:, 4]

# Group the data by elevation and retardation and count the number of rows per simulation
grouped_data = hit_sensor_data.groupby(['Elv', 'Retardation'])
simulation_counts = grouped_data.size().reset_index(name='Counts')

# Calculate collection efficiency per simulation
simulation_counts['Collection Efficiency'] = simulation_counts['Counts'] / 100.0  # this is the number of particles per simulation

# Get unique elevation and retardation values
elevation_values = np.unique(hit_sensor_data['Elv'])
retardation_values = np.unique(hit_sensor_data['Retardation'])

# Create plot_data DataFrame for storing the required data for plotting
plot_data = pd.DataFrame()

# Iterate over each elevation and retardation combination
for elevation in elevation_values:
    for retardation in retardation_values:
        # Check if there is a matching combination in simulation_counts
        if ((simulation_counts['Elv'] == elevation) & (simulation_counts['Retardation'] == retardation)).any():
            # Get the collection efficiency for the matching combination
            collection_efficiency = simulation_counts[
                (simulation_counts['Elv'] == elevation) & (simulation_counts['Retardation'] == retardation)][
                'Collection Efficiency'].values[0]
        else:
            collection_efficiency = np.nan

        # Filter data for the current elevation and retardation
        data_subset = hit_sensor_data[
            (hit_sensor_data['Elv'] == elevation) & (hit_sensor_data['Retardation'] == retardation)].copy()

        # Add the collection efficiency to the data subset
        data_subset['Collection Efficiency'] = collection_efficiency

        # Append the data subset to the plot_data DataFrame
        plot_data = pd.concat([plot_data, data_subset])

# Determine the number of plots per page
plots_per_page = 2

# Calculate the number of pages needed
num_pages = math.ceil(len(elevation_values) / plots_per_page)

# Create a PDF file to store the plots
pdf_file = '3d_scatter_plots.pdf'

with PdfPages(pdf_file) as pdf_pages:
    for page in range(num_pages):
        # Calculate the start and end indices for the current page
        start_index = page * plots_per_page
        end_index = min((page + 1) * plots_per_page, len(elevation_values))

        # Create a new figure for the current page
        fig = plt.figure(figsize=(8, 6 * (end_index - start_index)))
        fig.suptitle('3D Scatter Plots with Pass Energy vs. Time of Flight at Different Elevations')

        # Determine the overall range of the data
        min_pass_energy = plot_data['Pass Energy'].min()
        max_pass_energy = plot_data['Pass Energy'].max()
        min_time_of_flight = plot_data['Time of Flight'].min()
        max_time_of_flight = plot_data['Time of Flight'].max()

        # Get the common tick positions and labels for x, y, and z axes
        x_ticks = np.linspace(float(min_pass_energy), float(max_pass_energy), num=8)
        x_ticklabels = [f'{tick:.2f}' for tick in x_ticks]
        y_ticks = np.linspace(float(min_time_of_flight), float(max_time_of_flight), num=8)
        y_ticklabels = [f'{tick:.2f}' for tick in y_ticks]
        z_ticks = np.linspace(0.0, 1.0, num=10)
        z_ticklabels = [f'{tick:.2f}' for tick in z_ticks]

        for i, elevation_index in enumerate(range(start_index, end_index)):
            elevation = elevation_values[elevation_index]

            # Filter data for the current elevation
            elevation_data = plot_data[plot_data['Elv'] == elevation]

            # Convert columns to appropriate data types
            pass_energy = elevation_data['Pass Energy'].astype(float)
            time_of_flight = elevation_data['Time of Flight'].astype(float)
            collection_efficiency = elevation_data['Collection Efficiency'].astype(float)

            # Normalize the collection efficiency values between 0 and 1
            norm = Normalize(vmin=collection_efficiency.min(), vmax=collection_efficiency.max())
            normalized_collection_efficiency = norm(collection_efficiency)

            # Create 3D scatter plot
            ax = fig.add_subplot(end_index - start_index, 1, i + 1, projection='3d')
            scatter = ax.scatter(pass_energy, time_of_flight, collection_efficiency, c=normalized_collection_efficiency, cmap='autumn', s=50,
                       depthshade=True)

            # Set axis labels and title
            ax.set_xlabel('Pass Energy')
            ax.set_ylabel('Time of Flight')
            ax.set_zlabel('Collection Efficiency')
            ax.set_title(f'3D Scatter Plot at Elevation {elevation}')

            # Set the x-axis tick positions and labels
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels, fontsize=8)

            # Set the y-axis tick positions and labels
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticklabels, fontsize=8)

            # Set the z-axis tick positions and labels
            ax.set_zticks(z_ticks)
            ax.set_zticklabels(z_ticklabels, fontsize=8)


# %% (needs to be debugged, the color bars are not consistent in every graph)
        # Create a colorbar legend
            cbar_ax = fig.add_axes([0.05, 0.2, 0.02, 0.6])  # Adjust the position and size of the colorbar
            cbar = fig.colorbar(scatter, cax=cbar_ax)
            cbar.set_label('Collection Efficiency', fontsize=10)

        # Set the colorbar tick positions and labels
            cbar.set_ticks(z_ticks)
            cbar.set_ticklabels(z_ticklabels)

        # Save the current page to the PDF file
        pdf_pages.savefig(fig)

        # Close the current figure
        plt.close(fig)

# Show the plot
plt.show()


