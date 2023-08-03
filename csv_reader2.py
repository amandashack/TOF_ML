
#this code will read the data file given from the SIMION simulation and output the valuable
#data into an h5 fi

import pandas as pd

rows_to_skip = 10
data1 = pd.read_csv("C:\\Users\\lauren\\Documents\\Simion_Simulation\\simulation_files\\A0_E0_R100_maxKE1300_volt.csv", skiprows=rows_to_skip)

grouped_data = pd.concat([data1.iloc[::2].reset_index(drop=True), data1.iloc[1::2].reset_index(drop=True)], axis=1)

# Get the number of groups
num_groups = grouped_data.shape[1] // 2

# Group labels: initial (first 12) and final (last 12)
group_labels = ['initial'] * min(12, num_groups) + ['final'] * min(12, num_groups)

# Adjust column names for duplicate columns
grouped_data.columns = pd.MultiIndex.from_tuples([(col, label) for col, label in zip(grouped_data.columns, group_labels * 2)])

# Columns to remove
columns_to_remove = list(range(1, 9)) + list(range(12, 14)) + list(range(15, 17)) + list(range(20, 23))

# Remove specified columns
grouped_data = grouped_data.drop(grouped_data.columns[columns_to_remove], axis=1)

print(grouped_data)

#exports as a .csv file
output_file_path = "C:\\Users\\lauren\\Documents\\Simion_Simulation\\simulation_files\\A0_E0_R100_maxKE1300_volt_grouped.csv"
grouped_data.to_csv(output_file_path, index=False)
print("Data exported to:", output_file_path)

##export as a h5 file
#output_file_path = "C:\\Users\\mark\\Desktop\\Lauren_PythonDocs\\grouped_data.h5"
#grouped_data.to_hdf(output_file_path, key='data', mode='w')
#print("Data exported to:", output_file_path)
