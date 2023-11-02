# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 00:26:40 2023

@author: lauren
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import re
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU

# Defining the Y0 from the regression line on the NM simulation datasets
def y0_NM(x_value):
    y0 = -0.445 * x_value - 1.158
    return y0

# Function to create the dataframe for given folder path
def create_dataframe_from_files(folder_path):
        file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')] #and (re.search('A0_E', file)) and not (re.search('R090', file))]
        data = []

        for fname in file_list:
            temp = np.loadtxt('%s\%s' % (folder_path, fname), skiprows=2, delimiter=',')
            filename = os.path.splitext(fname)[0]  # Extract the filename without extension

            if len(data) == 0:
                data = np.hstack((temp, np.repeat(filename, temp.shape[0]).reshape(-1, 1)))
            else:
                data = np.concatenate((data, np.hstack((temp, np.repeat(filename, temp.shape[0]).reshape(-1, 1)))), axis=0)
                
        data = np.array(data)  # Convert the list to a NumPy array        
        valid_inds = np.where((data[:, 5].astype(float) > 402) & (np.abs(data[:, 6].astype(float)) < 13.7))
        valid_data = data[valid_inds]
        valid_data = pd.DataFrame(valid_data, columns=['Ion_number', 'Azimuth', 'Elevation', 'KE_initial', 'TOF', 'X_pos', 'Y_pos', 'Z_pos', 'KE_final', 'Filename'])
        valid_data['Filename'] = valid_data.iloc[:, -1].astype(str)  # Convert the filename column to string
        # Extract the retardation value from the 'Filename' column and convert to numeric
        retardation = [int(re.findall(r'_R(\d+)_', filename)[0]) for filename in valid_data['Filename']]
        valid_data['Retardation'] = retardation
        retardation = sorted(valid_data['Retardation'].unique())
        valid_data['Pass Energy'] = valid_data['KE_initial'].astype(float) - valid_data['Retardation'].astype(float)
        valid_data = valid_data[['Elevation', 'Retardation','Pass Energy', 'TOF']]

        # Create model data, narrow the data frame to only x inputs and output, also log2 necessary columns
        model_data = valid_data.copy()
        model_data['Pass Energy'] = np.log2(model_data['Pass Energy'].astype(float))
        model_data['TOF'] = np.log2(model_data['TOF'].astype(float))
        model_data = model_data.rename(columns={
            'Pass Energy': 'log2(Pass Energy)',
            'TOF': 'log2(TOF)'
        })

        return model_data



# test DATA
folder_path_test = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files\test_files\model_testing_2'
test_data = create_dataframe_from_files(folder_path_test)


#Filter out infinite values from test_data
test_data = test_data.replace([np.inf, -np.inf], np.nan)
test_data = test_data.dropna()


# Assign X and Y values for filtered test_data
X_test_data = test_data.iloc[:, 0:3].values.astype(float)
Y_test_data = test_data.iloc[:, -1].values.astype(float)  # Use 'Residuals' column as the target (output) variable

 # Evaluate the model on the filtered test_data
y0 = y0_NM(X_test_data[:,2])

#Calculating the residual
Y_test_data = Y_test_data - y0

#%%
# Load the trained model from the H5 file
model_filename = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files\EA_files\model_trained.h5'
loaded_model = load_model(model_filename)
#%%
loss1 = loaded_model.evaluate(X_test_data[0:930, :], Y_test_data[0:930], verbose=0)

loss2 = loaded_model.evaluate(X_test_data[1000:2000, :], Y_test_data[1000:2000], verbose=0)
#loss = np.float64(loss)

loss3 = loaded_model.evaluate(X_test_data[2100:3000, :], Y_test_data[2100:3000], verbose=0)

#%% 

Y_evaluated = loaded_model.predict(X_test_data)
plt.figure(101)
plt.plot(Y_test_data, 'blue')
plt.plot(Y_evaluated, 'red')

