# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:27:40 2023

@author: lauren
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

folder_path = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files'

file_list = [file for file in os.listdir(folder_path) if file.endswith('grouped.csv') and (re.search('A0_E0_R', file))]

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

# Convert the columns to numeric types
hit_sensor_data.iloc[:, 10] = pd.to_numeric(hit_sensor_data.iloc[:, 10], errors='coerce')
hit_sensor_data.iloc[:, 3] = pd.to_numeric(hit_sensor_data.iloc[:, 3], errors='coerce')

# Add 'pass energy' column
hit_sensor_data['Pass Energy'] = hit_sensor_data.iloc[:, 3] - hit_sensor_data.iloc[:, 10]

Y = np.log2(hit_sensor_data[df.columns[4]])
X = np.log2(hit_sensor_data.iloc[:, -1])  # New x-axis values

plt.plot(X, Y, '.', label='Data')

XX = np.ones((X.shape[0], 2))
XX[:, 1] = X
Xt = np.linalg.pinv(XX).T
th = np.dot(Y, Xt)
x = np.ones((10, 2))
x[:, 1] = np.linspace(X.min(), X.max(), num=10)
ypred = np.dot(th, x.T)
plt.plot(x[:, 1], ypred, '-', label=f'Best Fit Line: Y = {th[1]:.2f}X + {th[0]:.2f}')
plt.xlabel('Pass Energy')
plt.ylabel('log2(TOF)')
plt.legend()
plt.show()
