# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:24:37 2023

@author: lauren
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

folder_path = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files'
output_path = r'C:\Users\lauren\Desktop\NM_voltage_files'  # Updated output path

file_list = [file for file in os.listdir(folder_path) if file.endswith('grouped.csv') and (re.search('A0_E', file) )and not (re.search('E4',file))]

epass=[]
tof=[]
data=[]

for fname in file_list:
    print(fname)
    temp = np.loadtxt('%s\%s' % (folder_path, fname), skiprows=2, delimiter=',')
   
    if len(data) == 0:
        data = temp
    else:
        data = np.concatenate((data, temp), axis=0)
valid_inds=np.where((data[:,5]>402) * (np.abs(data[:,6])<13.7))
valid_data=data[valid_inds]    
 # Extract the retardation value from each file name and convert to numeric
valid_data['Retardation'] = valid_data['Filename'].apply(lambda x: int(re.findall(r'_R(\d+)_', x)[0]))
   # Add 'pass energy' column
valid_data['Pass Energy'] = valid_data.iloc[:, 3] - valid_data.iloc[:, 10]
epass=valid_data[:,11]
tof=valid_data[:,4] 
plt.plot(np.log2(epass),np.log2(tof),'.') 
plt.show()

#%%
combined_df = pd.concat(dfs, ignore_index=True)

combined_df[df.columns[5]] = pd.to_numeric(combined_df[df.columns[5]], errors='coerce')
combined_df[df.columns[6]] = pd.to_numeric(combined_df[df.columns[6]], errors='coerce')

# Convert 'Time of Flight' column to numeric type
combined_df[df.columns[4]] = pd.to_numeric(combined_df[df.columns[4]], errors='coerce')


#%%
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
    fig, ax = plt.subplots()
    
    
    for filename, group in groups:
       

        Y = np.log2(group[df.columns[4]])
        X = np.log2(group.iloc[:, -1])  # New x-axis value
        ax.plot(X, Y, '.', label='Data')

      
        XX = np.ones((X.shape[0], 2))
        XX[:, 1] = X
        Xt = np.linalg.pinv(XX).T
        th = np.dot(Y, Xt)
        
        x = np.ones((10, 2))
        x[:, 1] = np.linspace(2, 10, num=10)
        ypred = np.dot(th, x.T)
        plt.plot(x[:, 1], ypred, '-', label=f'Best Fit Line: Y = {th[1]:.2f}X + {th[0]:.2f}')
        plt.xlabel('log2(pass energy)')
        plt.ylabel('log2(TOF)')
        plt.legend()
        ax.legend()

        # Save graph as a PDF file
        graph_output_file = os.path.join(output_path, f"{filename.replace('.csv', '')}_fit_plot.pdf")
      

        # Create a new DataFrame with the specified columns and line of best fit equation
        new_df = group.copy()
        new_df['Line of Best Fit'] = f'Y = {th[1]:.2f}X + {th[0]:.2f}'

        # Save the new DataFrame as a CSV file
        new_df_output_file = os.path.join(output_path, f"{filename.replace('.csv', '')}_output_data.csv")
        new_df.to_csv(new_df_output_file, index=False)
    plt.show()
    with PdfPages(graph_output_file) as pdf:
       pdf.savefig(fig)
    plt.close(fig)  # Close the figure to free up memory
