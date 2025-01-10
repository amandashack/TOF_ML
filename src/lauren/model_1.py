# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:44:20 2023

@author: lauren
"""
#set up the model_data df
import pandas as pd
import numpy as np 
import os 
import re


#%%

#defining the Y0 from the regression line on the NM simulation datasets

def y0_NM(x_value):
    y0 = -0.4454*x_value - 1.154
    return y0

#%%


#make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)

#creating valid_data dataframe

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
valid_data['Pass Energy'] = valid_data['KE_initial'].astype(float) - valid_data['Retardation'].astype(float)
valid_data = valid_data[['Elevation', 'Retardation','Pass Energy', 'TOF']]

#create model data, narrow the data frame to only x inputs and output, also log2 necessary columns
model_data = valid_data.copy()
model_data['Pass Energy'] = np.log2(model_data['Pass Energy'].astype(float))
model_data['TOF'] = np.log2(model_data['TOF'].astype(float))
model_data = model_data.rename(columns={
    'Pass Energy': 'log2(Pass Energy)',
    'TOF': 'log2(TOF)'
})


#%%
#model_data_residual =  model_data.copy()
#model_data_residual['Residuals'] = model_data['log2(TOF)'] - y0_NM(model_data['log2(Pass Energy)'])
#%%

#assign X and Y values 
X=(model_data.iloc[:,0:3].values)
Y=(model_data.iloc[:,3].values)
X = X.astype(float)
Y = Y.astype(float)


#%% ONLY for testing residual

#assign X and Y values 
# X=(model_data_residual.iloc[:,0:3].values)
# Y=(model_data_residual.iloc[:,4].values)
# X = X.astype(float)
# Y = Y.astype(float)
#%% split the model_data into train and test data

from sklearn.model_selection import train_test_split
#create training and testing groups: 80%:20%, randomly chosen from model data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

import tensorflow as tf
X_train_tensor = tf.convert_to_tensor(X_train)
Y_train_tensor = tf.convert_to_tensor(Y_train)
X_test_tensor = tf.convert_to_tensor(X_test)
Y_test_tensor = tf.convert_to_tensor(Y_test)
 
#%% build and train the model

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense #allow for hidden layers

# Create the model
model = Sequential()

# Add the input layer
model.add(Dense(64, input_dim=3, activation='relu'))

# Add one or more hidden layers
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1, activation='linear')) 

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the testing data
loss = model.evaluate(X_test, Y_test, verbose=0)

# Print the loss  
print('Mean Squared Error (MSE) on test data: ', loss)

# Save the trained model
model.save('tof_prediction_model.h5')

#%%
#make predictions
import matplotlib.pyplot as plt

predictions= model.predict([X_test])
print(predictions)



plt.figure(102)
plt.plot(predictions, color='r', label='Predictions')
plt.plot(Y_test, color='b', label='Actual')
plt.xlabel('Data Point Indices')
plt.ylabel('Time of Flight (TOF)')
plt.legend()
plt.show()



#test both models(with and without residuals and then plot at different epochs)

