import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import re
from matplotlib import cm
from matplotlib.ticker import FixedLocator

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
hit_sensor_data['Retardation'] = hit_sensor_data['Filename'].apply(
    lambda x: int(re.findall(r'_R(\d+)_', x)[0]) if re.findall(r'_R(\d+)_', x) else None
)

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface using plot_trisurf
tri = ax.plot_trisurf(x_data, y_data, z_data, linewidth=0.1, cmap='gist_rainbow', vmin=tof_data.min(), vmax=tof_data.max())

# Add a colorbar
cbar = fig.colorbar(tri, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Time of Flight')

# Set labels for each axis
ax.set_xlabel('Elevation')
ax.set_ylabel('Kinetic Energy')
ax.set_zlabel('Retardation')

# Save the figure as 'teste.pdf'
plt.savefig('teste.pdf')

# Display the plot
plt.show()
