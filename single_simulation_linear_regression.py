# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:27:40 2023

@author: lauren
"""
#this code opens chosen .csv files, filters the data to keep only the electrons that hit the sensor
#and graphs their TOFs vs. pass energy on a log scale. Then, it graphs a line of best fit 
#the plot will be outputted as a .pdf file  and the data with the fit parameter will be 
#outputted as a .csv file
#################################################################################################
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

folder_path = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files'

file_list = [file for file in os.listdir(folder_path) if file.endswith('grouped.csv') and (re.search('A0_E1_R1000', file))]

for file in file_list:
    dfs = []

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
    hit_sensor_data.iloc[:, 10] = pd.to_numeric(hit_sensor_data.iloc[:, 10], errors='coerce').values
    hit_sensor_data.iloc[:, 3] = pd.to_numeric(hit_sensor_data.iloc[:, 3], errors='coerce').values

    # Add 'pass energy' column
    hit_sensor_data['Pass Energy'] = hit_sensor_data.iloc[:, 3] - hit_sensor_data.iloc[:, 10]

    groups = hit_sensor_data.groupby('Filename')

for filename, group in groups:
    fig, ax = plt.subplots()

    X = np.log2(group.iloc[:, 4])  # Use group data
    Y = np.log2(group.iloc[:, 11])  # Use group data
    plt.plot(X, Y, '.', label='Data')

    XX = np.ones((X.shape[0], 2))
    XX[:, 1] = X
    Xt = np.linalg.pinv(XX).T
    th = np.dot(Y, Xt)
    x = np.ones((X.shape[0], 2))  # Use X.shape[0] for the same number of points
    x[:, 1] = X
    ypred = np.dot(th, x.T)
    plt.plot(x[:, 1], ypred, '-', label=f'Best Fit Line: Y = {th[1]:.2f}X + {th[0]:.2f}')
    plt.xlabel('log2(Pass Energy)')
    plt.ylabel('log2(TOF)')
    plt.legend()

    # Set plot limits to make sure the line reaches all points
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))

    # Save graph as a PDF file
    graph_output_file = f"{filename.replace('.csv', '')}_plot1.pdf"
    with PdfPages(graph_output_file) as pdf:
        pdf.savefig(fig)

    # Create a new DataFrame with the specified columns and line of best fit equation
    new_df = group.copy()
    new_df['Line of Best Fit'] = f'Y = {th[1]:.2f}X + {th[0]:.2f}'

    # Save the new DataFrame as a CSV file
    new_df_output_file = f"{filename.replace('.csv', '')}_new_data1.csv"
    new_df.to_csv(new_df_output_file, index=False)
    plt.show()

    plt.close(fig)  # Close the figure to free up memory
