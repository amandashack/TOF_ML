"""
Created on Oct 3 2023
this code should access the raw SIMION simulation .csv files essentially clean them up
so data analysis for later will be easier.

enter the folder path in which all your raw SIMION .csv files are located into
the command line, input first and output second. Skipped lines are third.
If you would like the csv files to be converted to h5 files, add a fourth command line input that says h5


@author: Amanda & Lauren
"""
import os

import h5py
import pandas as pd
import sys

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


def process_multi_csv(input_dir, output_dir, rows_to_skip, file_type="csv"):
    """
    Process all CSV files in the input directory

    Parameters:
        input_dir: input directory where CSV files live
        output_dir: where the new CSV files will go
        rows_to_skip: a list of rows to ignore in the CSV files (assumed they are all the same)
    """
    for file_name in os.listdir(input_dir):
        # make sure it's a csv file
        if file_name.endswith(".csv"):  # this will be specific to what you name your files
            input_file_path = os.path.join(input_dir, file_name)
            # Get the file name and extension separately
            file_name_only, file_extension = os.path.splitext(file_name)
            grouped_data = process_csv_file(input_file_path, rows_to_skip)
            if file_type == "csv":
                output_file_name = f"{file_name_only}_grouped{file_extension}"
                output_file_path = os.path.join(output_dir, output_file_name)
                grouped_data.to_csv(output_file_path, index=False)
            elif file_type == "h5":
                output_file_name = f"{file_name_only}_grouped.h5"
                output_file_path = os.path.join(output_dir, output_file_name)
                f = h5py.File(output_file_path, "w")
                g1 = f.create_group("data1")
                g1.create_dataset("ion_number", data=grouped_data.loc[:, "Ion N"]["initial"])
                g1.create_dataset("azimuth", data=grouped_data.loc[:, "Azm"]["initial"])
                g1.create_dataset("elevation", data=grouped_data.loc[:, "Elv"]["initial"])
                g1.create_dataset("initial_ke", data=grouped_data.loc[:, "KE"]["initial"])
                g1.create_dataset("tof", data=grouped_data.loc[:, "TOF"]["final"])
                g1.create_dataset("x", data=grouped_data.loc[:, "X"]["final"])
                g1.create_dataset("y", data=grouped_data.loc[:, "Y"]["final"])
                g1.create_dataset("z", data=grouped_data.loc[:, "Z"]["final"])
                g1.create_dataset("final_ke", data=grouped_data.loc[:, "KE"]["final"])
                g1.create_dataset("all_data", data=grouped_data.loc[:, :])
            print("Data exported to:", output_file_path)


if __name__ == "__main__":
    print(sys.argv[3])
    if len(sys.argv) < 2:
        print("must indicate input directory location and output directory "
              "location at a minimum.")
    elif len(sys.argv) >2 and len(sys.argv) < 6:
        # Directory containing CSV files
        input_dir = os.path.abspath(sys.argv[1])
        # Output directory for the processed files
        output_dir = os.path.abspath(sys.argv[2])
        # rows in the csv files to skip
        rows_to_skip = int(sys.argv[3])
        file_type = "csv"
        if len(sys.argv) == 5:
            file_type = sys.argv[4]
        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        process_multi_csv(input_dir, output_dir, rows_to_skip, file_type=file_type)
    elif len(sys.argv) > 5:
        print("too many command line arguments")