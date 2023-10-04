# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:35:11 2023

@author: lauren
"""
import pandas as pd
import numpy as np 
import os 
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU

# Define Swish activation function
def swish(x):
    return x * tf.sigmoid(x)

# Defining the Y0 from the regression line on the NM simulation datasets
def y0_NM(x_value):
    y0 = -0.4454 * x_value - 1.154
    return y0

# Function to create the dataframe for given folder path
def create_dataframe_from_files(folder_path):
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
   
# Paths for model data and test data
folder_path_model = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files'
folder_path_test = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files\test_files'

# Create model_data and test_data dataframes
model_data = create_dataframe_from_files(folder_path_model)
test_data = create_dataframe_from_files(folder_path_test)

print("Model Data:")
print(model_data.head())
print("\nTest Data:")
print(test_data.head())
#%%
model_data_residual = model_data.copy()
model_data_residual['Residuals'] = model_data['log2(TOF)'] - y0_NM(model_data['log2(Pass Energy)'])

# Assign X and Y values for model_data
X_model_data = model_data_residual.iloc[:, 0:3].values.astype(float)
Y_model_data = model_data_residual.iloc[:, -1].values.astype(float)  # Use 'Residuals' column as the target (output) variable

# Assign X and Y values for test_data
X_test_data = test_data.iloc[:, 0:3].values.astype(float)
Y_test_data = test_data.iloc[:, -1].values.astype(float)  # Use 'Residuals' column as the target (output) variable



#%%

# Define a list to store the results for different epochs
epochs_list = [5] * 20

## Create a PDF to store the plots
#pdf_filename = 'tof_prediction_plots_residual.pdf'
#pdf_pages = PdfPages(pdf_filename)

model = Sequential()

model.add(Dense(32, input_dim=3))
model.add(LeakyReLU(alpha=0.001))
model.add(Dense(16))
model.add(LeakyReLU(alpha=0.001))
model.add(Dense(16, activation=swish))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

loss_list = []

for epochs in epochs_list:
    # Train the model with all data from model_data
    model.fit(X_model_data, Y_model_data, epochs=epochs, batch_size=16, verbose=0)
    
    # Evaluate the model on the test_data
    loss = model.evaluate(X_test_data, Y_test_data -y0_NM(test_data['log2(Pass Energy)']), verbose=0)
    print('Mean Squared Error (MSE) on test data:', loss)
    loss_list.append(loss)
    
    # Make predictions on test_data
    predictions = model.predict(X_test_data)
    
plt.plot(loss_list, ".")
plt.show()
