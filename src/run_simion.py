# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:38:31 2023

@author: lauren
"""

import os
import numpy as np
import argparse
import re
import h5py
from ltspice_runner import modify_cir_file, run_simulation, check_currents
from voltage_generator import calculateVoltage_NelderMeade


def generate_files(data, output_dir):
    with h5py.File(output_dir, "w") as f:
        g1 = f.create_group("data1")
        g1.create_dataset("ion_number", data=data['Ion number'])
        g1.create_dataset("initial_ke", data=data['KE_initial'])
        g1.create_dataset("initial_ke_error", data=data['KE_initial_error'])
        g1.create_dataset("x", data=data['X'])
        g1.create_dataset("y", data=data['Y'])
        g1.create_dataset("tof", data=data['TOF'])
        g1.create_dataset("elevation", data=data['Elv'])
        g1.create_dataset("final_ke", data=data['KE_final'])
        g1.create_dataset("final_ke_error", data=data['KE_final_error'])
        print("Data exported to:", output_dir)


# Function to parse the log file and extract required information into CSV
def parse_log_to_csv(input_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Relevant information starts from line 12 (index 11)
    current_ion = {"Ion number": [], "KE_initial": [], "KE_initial_error": [],
                   "TOF": [], "X": [], "Y": [],
                   "Elv": [], "KE_final": [], "KE_final_error": []}
    for i, line in enumerate(lines[11:], 11):
        if 'Event(Ion Created)' in line:
            ion_number = re.search(r'Ion\((\d+)\)', line)
            current_ion['Ion number'].append(int(ion_number.group(1)))
            ke = re.search(r'KE\(([\d\.]+)', lines[i+2])
            current_ion['KE_initial'].append(float(ke.group(1)))
            ke = re.search(r'KE_Error\(([\d\.]+)', lines[i + 2])
            current_ion['KE_initial_error'].append(float(ke.group(1)))
            TOF = re.search(r'TOF\((-?[\d\.]+)', lines[i+4])
            x_pos = re.search(r'X\((-?[\d\.]+)', lines[i+5])
            y_pos = re.search(r'Y\((-?[\d\.]+)', lines[i+5])
            try:
                elv = re.search(r'Elv\((-?[\d\.]+)', lines[i+6])
                current_ion['Elv'].append(float(elv.group(1)))
            except AttributeError as e:
                try:
                    elv = re.search(r'Elv\((-?[\d\.]+)', lines[i+5])
                    current_ion['Elv'].append(float(elv.group(1)))
                except AttributeError as e:
                    print("could not find elevation on either line!")
            current_ion['TOF'].append(float(TOF.group(1)))
            current_ion['X'].append(float(x_pos.group(1)))
            current_ion['Y'].append(float(y_pos.group(1)))
            ke = re.search(r'KE\((-?[\d\.]+)', lines[i+6])
            current_ion['KE_final'].append(float(ke.group(1)))
            ke = re.search(r'KE_Error\(([\d\.]+)', lines[i+6])
            current_ion['KE_final_error'].append(float(ke.group(1)))
    base_name = os.path.basename(input_file_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}.h5"
    output_file_path = os.path.join(os.path.dirname(input_file_path), output_filename)
    generate_files(current_ion, output_file_path)
    return


# generate a .fly2 file type.  file name must be provided as directory string INCLUDING the .FLY2 post-fix
def generate_fly2File_lognorm(filenameToWriteTo, numParticles=100, medianEnergy=10, energySigma=1):

    # Define path for simulation files and change directory
    simion_files_path = "../simulations/simionSimulationFiles"
    os.chdir(simion_files_path)

    # Check if .fly2 file with this name already exists
    fileExists = os.path.isfile(filenameToWriteTo)

    # Delete previous copy, if there is one
    if fileExists:
        os.remove(filenameToWriteTo)

    # Open up file to write to
    with open(filenameToWriteTo, "w") as fileOut:
        # Write the Lua function for lognormal distribution at the beginning of the file
        fileOut.write("-- Define lognormal distribution function\n")
        fileOut.write("local math = require('math')\n")
        fileOut.write("function lognormal(median, sigma)\n")
        fileOut.write("  return function()\n")
        fileOut.write("    local z = math.log(median) - 0.5 * sigma^2\n")
        fileOut.write("    local x = z + sigma * math.sqrt(-2 * math.log(math.random()))\n")
        fileOut.write("    return math.exp(x)\n")
        fileOut.write("  end\n")
        fileOut.write("end\n\n")

        # Now define the particle distribution using the lognormal function
        fileOut.write("particles {\n")
        fileOut.write("  coordinates = 0,\n")
        fileOut.write("  standard_beam {\n")
        fileOut.write(f"    n = {numParticles},\n")
        fileOut.write("    tob = 0,\n")
        fileOut.write("    mass = 0.000548579903,\n")
        fileOut.write("    charge = -1,\n")
        fileOut.write(f"    ke = distribution(lognormal({medianEnergy}, {energySigma}))\n")
        fileOut.write("    az =  0\n")
        fileOut.write("    el =  uniform_distribution {\n")
        fileOut.write("      min = -5,\n")
        fileOut.write("      max = 5\n")
        fileOut.write("    },\n")
        fileOut.write("    cwf = 1,\n")
        fileOut.write("    color = 0,\n")
        fileOut.write("    position =  sphere_distribution {\n")
        fileOut.write("      center = vector(244, 0, 0),\n")
        fileOut.write("      radius = 0,\n")
        fileOut.write("      fill = true")
        fileOut.write("    }\n")
        fileOut.write("  }\n")
        fileOut.write("}")


# helper method that makes the call to run simion to fly particles.  This method links up all the required file
# directories into a single call to simion, which will then fly the simulation and create an outputFile
# fly2FileDir contains the definition of the particles that should be simulated.
# This file is made with the method 'makeFLY2File()'
# outputFile is the directory to which the output log will be saved.  This log is a summary of the simulation
# results, and is made by simion as simion runs the simulation
# recordingFile is a special file that has the recording options.  This is a file that can only be made in simion -
# it is a binary file that has a bunch of flags to tell the program what to record during simulation.
# the iobFileLoc is directory to the .IOB file, which is the ion bench file.  I am not fully sure I understand what
# this is, but I think it is a file that links the potential arrays to the simulation.
def runSimion(fly2FileDir, luaFile, voltage_array, outputFile, recordingFile, iobFileLoc, baseDir):
    # Check if outputFile exists and delete it if it does
    if os.path.isfile(outputFile):
        os.remove(outputFile)

    # Convert voltage_array to a space-separated string
    voltage_string = ' '.join(map(str, voltage_array))

    # Command parts may need to be individually quoted if they contain spaces
    luaCommand = f"--lua @{luaFile} {voltage_string}"
    flyCommand = f"fly --recording-output=\"{outputFile}\" --recording=\"{recordingFile}\" " \
                 f"--particles=\"{fly2FileDir}\" --restore-potentials=1 \"{iobFileLoc}\""

    # Construct the full command
    fullCommand = f"simion.exe --nogui {flyCommand} {luaCommand}"

    # Change to the SIMION working directory
    original_cwd = os.getcwd()
    os.chdir(r"C:\Users\proxi\Downloads\Simion_8-1-20230825T223627Z-001\Simion_8-1")

    # Execute the command
    os.system(fullCommand)
    # Go back to the base directory and delete temporary files
    os.chdir(baseDir)
    os.system('del *.tmp')
    os.chdir(original_cwd)


if __name__ == '__main__':
    ltspice_path = "C:\\Users\\proxi\\AppData\\Local\\Programs\\ADI\\LTspice\\LTspice.exe"
    baseDir = "C:/Users/proxi/Documents/SimionRunFiles"
    iobFileLoc = baseDir + "/workbench.iob"
    recordingFile = baseDir + "/recordingOptions.rec"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    lua_path = dir_path + "\\lua_script.lua"

    parser = argparse.ArgumentParser(description='code for running LTSpice simulations')

    # Add arguments
    parser.add_argument('retardation', metavar='N', type=int, help='Required retardation value')
    parser.add_argument('simulation_dir', metavar='N', type=int, help='Required output directory')
    parser.add_argument('--front_voltage', type=float, help='Optional front voltage value')
    parser.add_argument('--back_voltage', type=float, help='Optional back voltage value')

    args = parser.parse_args()

    Fly2File = args.simulation_dir + "\\particles.fly2"
    simion_output_path = args.simulation_dir + "\\simion_output\\flightOutputLog.txt"

    cir_filepath = dir_path + "\\voltage_divider.cir"
    new_cir_filepath = args.simulation_dir + f"\\ltspice\\voltage_divider_{args.retardation}.cir"
    raw_file_path = args.simulation_dir + f"\\ltspice\\voltage_divider_{args.retardation}.raw"

    ltspice_out_dir = args.simulation_dir + "\\ltspice"

    # Modify the .cir file with new voltages
    new_voltages, resistor_values = calculateVoltage_NelderMeade(args.retardation,
                                                                 args.front_voltage,
                                                                 args.back_voltage)
    modify_cir_file(cir_filepath, new_voltages, new_cir_filepath)

    # Run the LTspice simulation
    run_simulation(ltspice_path, new_cir_filepath, ltspice_out_dir)

    # Read the .raw file and check current values
    ok, max_val = check_currents(raw_file_path)
    if ok:
        runSimion(Fly2File, lua_path, new_voltages, simion_output_path, recordingFile, iobFileLoc, baseDir)
        parse_log_to_csv(simion_output_path)
