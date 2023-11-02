"""
Created on Oct 3 2023
this code should access the grouped SIMION simulation .csv files to convert them to h5 files

enter the folder path in which all your grouped SIMION .csv files are located into
the command line, input first and output second.


@author: Amanda & Lauren
"""
import os

import h5py
import pandas as pd
import sys


def process_multi_csv(input_dir, output_dir):
    """
    Process all CSV files in the input directory

    Parameters:
        input_dir: input directory where CSV files live
        output_dir: where the new CSV files will go
        rows_to_skip: a list of rows to ignore in the CSV files (assumed they are all the same)
    """
    for file_name in os.listdir(input_dir):
        # make sure it's a csv file
        if file_name.endswith("grouped.csv"):  # this will be specific to what you name your files
            input_file_path = os.path.join(input_dir, file_name)
            # Get the file name and extension separately
            file_name_only, file_extension = os.path.splitext(file_name)
            grouped_data = pd.read_csv(input_file_path, header=[0, 1])
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
    if len(sys.argv) < 2:
        print("must indicate input directory location and output directory "
              "location at a minimum.")
    elif len(sys.argv) >1 and len(sys.argv) < 4:
        # Directory containing CSV files
        input_dir = os.path.abspath(sys.argv[1])
        # Output directory for the processed files
        output_dir = os.path.abspath(sys.argv[2])
        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        process_multi_csv(input_dir, output_dir)
    elif len(sys.argv) > 3:
        print("too many command line arguments")