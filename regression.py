import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Setting up hit_sensor_data
folder_path = r'C:\Users\lauren\Documents\Simion_Simulation\simulation_files'  # file path
data = []

file_list = [file for file in os.listdir(folder_path) if (re.search('A0_E', file) and re.search('grouped', file))]  # this will differ

for fname in file_list:
    temp = np.loadtxt('%s\%s' % (folder_path, fname), skiprows=2, delimiter=',')
   
    if len(data) == 0:
        data = temp
    else:
        data = np.concatenate((data, temp), axis=1)

X = np.log2(data[:, 3])
Y = np.log2(data[:, 4])
plt.plot(X, Y, '.', label='Data')

XX = np.ones((X.shape[0], 2))
XX[:, 1] = X
Xt = np.linalg.pinv(XX).T
th = np.dot(Y, Xt)
x = np.ones((10, 2))
x[:, 1] = np.linspace(2, 10, num=10)
ypred = np.dot(th, x.T)
plt.plot(x[:, 1], ypred, '-', label=f'Best Fit Line: Y = {th[1]:.2f}X + {th[0]:.2f}')
plt.xlabel('log2(KE)')
plt.ylabel('log2(TOF)')
plt.legend()

# Create a new array with the line of best fit equation appended
data_with_equation = np.hstack((data, np.array([[f'Y = {th[1]:.10f}X + {th[0]:.10f}']] * data.shape[0])))

# Save the data with the line of best fit equation as a CSV file
csv_file_path = os.path.join(folder_path, 'data_with_equation.csv')
np.savetxt(csv_file_path, data_with_equation, delimiter=',', fmt='%s')

# Save the plot as a PDF file
plot_file_path = os.path.join(folder_path, 'plot_name.pdf')
plt.savefig(plot_file_path, format='pdf')

plt.show()


