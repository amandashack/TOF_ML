# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:30:21 2023

@author: lauren

this code filters all the data provided in the folder path and plots log2(Pass Energy)
vs. log2(TOF) and creates a best fit line and equation of the line in the legend. 
The points are also color coded based off of retardation value
"""
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os

folder_path = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files'
output_path = r'C:\Users\lauren\Desktop\NM_voltage_files\combined-regression'

file_list = [file for file in os.listdir(folder_path) if file.endswith('grouped.csv') and (re.search('A0_E', file)) and not (re.search('E4', file))]

data = []

for fname in file_list:
    temp = np.loadtxt('%s\%s' % (folder_path, fname), skiprows=2, delimiter=',')
    filename = os.path.splitext(fname)[0]  # Extract the filename without extension

    if len(data) == 0:
        data = np.hstack((temp, np.repeat(filename, temp.shape[0]).reshape(-1, 1)))
    else:
        data = np.concatenate((data, np.hstack((temp, np.repeat(filename, temp.shape[0]).reshape(-1, 1)))), axis=0)

valid_inds = np.where((data[:, 5].astype(float) > 402) & (np.abs(data[:, 6].astype(float)) < 13.7))  # Convert relevant columns to float for element-wise comparison
valid_data = data[valid_inds]

valid_data = pd.DataFrame(valid_data, columns=['Ion_number', 'Azimuth', 'Elevation', 'KE_initial', 'TOF', 'X_pos', 'Y_pos', 'Z_pos', 'KE_final', 'Filename'])

valid_data['Filename'] = valid_data.iloc[:, -1].astype(str)  # Convert the filename column to string

# Extract the retardation value from the 'Filename' column and convert to numeric
retardation = [int(re.findall(r'_R(\d+)_', filename)[0]) for filename in valid_data['Filename']]
valid_data['Retardation'] = retardation

retardation = sorted(valid_data['Retardation'].unique())
color_range = np.linspace(0, 1, len(retardation))
colors = plt.cm.autumn(color_range)

# Create a color dictionary to map retardation values to colors
color_dict = {value: color for value, color in zip(retardation, colors)}

X = pd.to_numeric(valid_data.iloc[:, 3]) - pd.to_numeric(valid_data.iloc[:, 10])  # Convert epass column to numeric
Y = pd.to_numeric(valid_data.iloc[:, 4])  # Convert tof column to numeric

# Plot each point with the assigned color based on retardation value
for r in retardation:
    mask = valid_data['Retardation'] == r
    plt.plot(np.log2(X[mask]), np.log2(Y[mask]), '.', color=color_dict[r], label=f'Retardation: {r}', marker='o', markersize=2)

plt.xlabel('log2(Pass Energy)')
plt.ylabel('log2(TOF)')

XX = np.ones((X.shape[0], 2))
XX[:, 1] = np.log2(X)  # Use log2(X) instead of X for best fit line
Xt = np.linalg.pinv(XX).T
th = np.dot(np.log2(Y), Xt)  # Use log2(Y) instead of Y for best fit line

x = np.ones((10, 2))
x[:, 1] = np.linspace(np.log2(X).min(), np.log2(X).max(), num=10)  # Use log2(X) range for x values
ypred = np.dot(th, x.T)
plt.plot(x[:, 1], ypred, '-', label=f'Best Fit Line: Y = {th[1]:.4g}X + {th[0]:.4g}')
plt.xlabel('log2(Pass Energy)') 
plt.ylabel('log2(TOF)')
plt.legend(prop={'size': 6})  # Adjust the font size of the legend

figure_output_file = os.path.join(output_path, 'combined_plot.pdf')
plt.savefig(figure_output_file)
plt.show()

