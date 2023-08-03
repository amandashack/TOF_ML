# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:29:49 2023

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
model_data_residual =  model_data.copy()
model_data_residual['Residuals'] = model_data['log2(TOF)'] - y0_NM(model_data['log2(Pass Energy)'])

#%% residual testingb

#assign X and Y values 
X=(model_data_residual.iloc[:,0:3].values)
Y=(model_data_residual.iloc[:,4].values)
X = X.astype(float)
Y = Y.astype(float)
#%% split the model_data into train and test data

#create training and testing groups: 80%:20%, randomly chosen from model data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

X_train_tensor = tf.convert_to_tensor(X_train)
Y_train_tensor = tf.convert_to_tensor(Y_train)
X_test_tensor = tf.convert_to_tensor(X_test)
Y_test_tensor = tf.convert_to_tensor(Y_test)
 
# Define a list to store the results for different epochs
#epochs_list = range(5, 15, 10)  
epochs_list = [5]*20
# Create a PDF to store the plots
pdf_filename = 'tof_prediction_plots_residual.pdf'
pdf_pages = PdfPages(pdf_filename)

 # Create the model
model = Sequential()
model.add(Dense(32, input_dim=3, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(12, activation='relu'))  
model.add(Dense(1, activation='linear')) 
model.compile(loss='mean_squared_error', optimizer='adam')

#for adam, learning rate is has the default value of .001
#learning_rate = 0.001
loss_list=[]
# Create a loop to train and plot the model for each number of epochs
for epochs in epochs_list:
    print(f"Training the model with {epochs} epochs; Residual Method")
    
    # Train the model
    model.fit(X_train, Y_train, epochs=epochs, batch_size=32, verbose=0)
    
    # Evaluate the model on the testing data

    loss = model.evaluate(X_test, Y_test, verbose=0)
    loss_list+=[loss]
    print('Mean Squared Error (MSE) on test data: ', loss)
    
  # Make predictions
    predictions = model.predict(X_test)

    # Plot the predictions and actual values
    plt.figure()
    plt.plot(predictions, color='r', label=f'Predictions (MSE: {loss:.7f})')
    plt.plot(Y_test, color='b', label='Actual')
    plt.xlabel('Data Point Indices')
    plt.ylabel('Residuals [log2(TOF)-Y0(log2(Pass Energy))]')
    # Set y-axis limits to ensure consistent tick marks
    y_min = min(np.min(predictions), np.min(Y_test))
    y_max = max(np.max(predictions), np.max(Y_test))
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.title(f'Model with {epochs} epochs: Residual Method')
    plt.figtext(0.95, 0.05, f'Epochs: {epochs}', ha='right', fontsize=8)
    
    # Save the plot to the PDF file
    pdf_pages.savefig()
    plt.close()
print(loss_list)
# Close the PDF after saving all the plots
pdf_pages.close()