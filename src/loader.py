"""
Created on Oct 3 2023
this code should access the raw SIMION simulation .csv files essentially clean them up
so data analysis for later will be easier.

enter the folder path in which all your raw SIMION .csv files are located into "input_directory"
as well as an "output_directory location"


@author: Amanda
"""
import os
import pandas as pd


# Function to process each CSV file
def process_csv_file(file_path, rows_to_skip):
    rows_to_skip = rows_to_skip
    data1 = pd.read_csv(file_path, skiprows=rows_to_skip)

    grouped_data = pd.concat([data1.iloc[::2].reset_index(drop=True), data1.iloc[1::2].reset_index(drop=True)], axis=1)

    # Get the number of groups
    num_groups = grouped_data.shape[1] // 2

    # Group labels: initial (first 12) and final (last 12)
    group_labels = ['initial'] * min(12, num_groups) + ['final'] * min(12, num_groups)

    # Adjust column names for duplicate columns
    grouped_data.columns = pd.MultiIndex.from_tuples(
        [(col, label) for col, label in zip(grouped_data.columns, group_labels * 2)])

    # Columns to remove
    columns_to_remove = list(range(1, 9)) + list(range(12, 14)) + list(range(15, 17)) + list(range(20, 23))

    # Remove specified columns
    grouped_data = grouped_data.drop(grouped_data.columns[columns_to_remove], axis=1)

    return grouped_data


# Directory containing CSV files
input_directory = "C:\\Users\\lauren\\Documents\\Simion_Simulation\\SimionRunFiles"

# Output directory for the processed files
output_directory = "C:\\Users\\lauren\\Documents\\Simion_Simulation\\simulation_files\\EA_files"

# Create the output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

# Process all CSV files in the input directory
for file_name in os.listdir(input_directory):
    if file_name.endswith("1200.csv"):  # this will be specific to what you name your files
        input_file_path = os.path.join(input_directory, file_name)

        # Get the file name and extension separately
        file_name_only, file_extension = os.path.splitext(file_name)

        grouped_data = process_csv_file(input_file_path)
        output_file_name = f"{file_name_only}_grouped{file_extension}"
        output_file_path = os.path.join(output_directory, output_file_name)

        grouped_data.to_csv(output_file_path, index=False)

        print("Data exported to:", output_file_path)
