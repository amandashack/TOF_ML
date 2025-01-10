# -*- coding: utf-8 -*-
"""
Created on Jan 30 2014

@author: = Amanda
"""

import subprocess
import os
import argparse
import h5py
import csv
import sys
sys.path.insert(0, os.path.abspath('../..'))
from tof_ml.simulations.voltage_generator import calculateVoltage_NelderMeade
import time
import numpy as np


def check_h5(file):
    if file.endswith("h5"):
        with h5py.File(file, 'r') as f:
            x_tof = f['data1']['x'][:]
            y_tof = f['data1']['y'][:]
            tof_values = f['data1']['tof'][:]
            initial_ke = f['data1']['initial_ke'][:]
            elevation = f['data1']['elevation'][:]
            print(initial_ke)


def generate_files_old(data, output_dir):
    with h5py.File(output_dir, "w") as f:
        g1 = f.create_group("data1")
        g1.create_dataset("ion_number", data=data['Ion number'])
        g1.create_dataset("initial_ke", data=data['KE_initial'])
        #g1.create_dataset("initial_ke_error", data=data['KE_initial_error'])
        g1.create_dataset("x", data=data['X'])
        g1.create_dataset("y", data=data['Y'])
        g1.create_dataset("tof", data=data['TOF'])
        g1.create_dataset("elevation", data=data['Elv'])
        g1.create_dataset("final_ke", data=data['KE_final'])
        g1.create_dataset("final_ke_error", data=data['KE_final_error'])
        print("Data exported to:", output_dir)


def generate_files(data, output_dir):
    output_file, ext = os.path.splitext(output_dir)
    output_file = output_file + ".h5"
    with h5py.File(output_file, "w") as f:
        g1 = f.create_group("data1")
        for key, values in data.items():
            g1.create_dataset(key, data=values)
        print("Data exported to:", output_dir)


def parse_and_process_data(input_file_path):
    with open(input_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header line if there is one, or adjust if data starts immediately

        data = {
            "ion_number": [], "initial_azimuth": [], "initial_elevation": [], "initial_ke": [],
            "tof": [], "x": [], "y": [], "z": [], "final_azimuth": [], "final_elevation": [], "final_ke": []
        }

        for start_line in reader:
            if not start_line:
                continue
            splat_line = next(reader, None)  # Read the subsequent line for 'splat' data

            # Start line processing (initial conditions)
            data["ion_number"].append(int(start_line[0]))
            data["initial_azimuth"].append(float(start_line[8]))
            data["initial_elevation"].append(float(start_line[9]))
            data["initial_ke"].append(float(start_line[10]))

            # Splat line processing (final conditions)
            if splat_line:
                data["tof"].append(float(splat_line[2]))
                data["x"].append(float(splat_line[5]))
                data["y"].append(float(splat_line[6]))
                data["z"].append(float(splat_line[7]))
                data["final_azimuth"].append(float(splat_line[8]))
                data["final_elevation"].append(float(splat_line[9]))
                data["final_ke"].append(float(splat_line[10]))

        generate_files(data, input_file_path)


def generate_fly2File(filenameToWriteTo, numParticles=500, minEnergy=10, maxEnergy=0):
    #check if .fly2 file with this name already exists.
    fileExists = os.path.isfile(filenameToWriteTo)
    #delete previous copy, if there is one
    if fileExists == True:
        os.remove(filenameToWriteTo)

    #open up file to write to
    fileOut = open(filenameToWriteTo, "w")

    #write out the .fly2 scripts
    fileOut.write("particles {\n")
    fileOut.write("  coordinates = 0,\n")
    fileOut.write("  standard_beam {\n")
    fileOut.write("    n = " + str(numParticles) + ",\n")
    fileOut.write("    tob = 0,\n")
    fileOut.write("    mass = 0.000548579903,\n")
    fileOut.write("    charge = -1,\n")
    fileOut.write("    ke =  uniform_distribution {\n")
    fileOut.write("      min = " + str(minEnergy) + ",\n")
    fileOut.write("      max = " + str(maxEnergy) + "\n")
    fileOut.write("    },\n")
    fileOut.write("    az =  single_value {0},\n")
    fileOut.write("    el =  uniform_distribution {\n")
    fileOut.write("      min = 0,\n")
    fileOut.write("      max = 3\n")
    fileOut.write("    },\n")
    fileOut.write("    cwf = 1,\n")
    fileOut.write("    color = 0,\n")
    fileOut.write("    position =  sphere_distribution {\n")
    fileOut.write("      center = vector(24.4, 0, 0),\n")
    fileOut.write("      radius = 2,\n")
    fileOut.write("      fill = true")
    fileOut.write("    }\n")
    fileOut.write("  }\n")
    fileOut.write("}")

    #close file
    fileOut.close()

# generate a .fly2 file type.  file name must be provided as directory string INCLUDING the .FLY2 post-fix
def generate_fly2File_lognorm(filenameToWriteTo, min_energy, max_energy, numParticles=100, medianEnergy=10.,
                              energySigma=1., shift=0., max_angle=3):

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
        fileOut.write("function lognormal(median, sigma, shift, min, max)\n")
        fileOut.write("  return function()\n")
        fileOut.write("    local value\n")
        fileOut.write("    repeat\n")
        fileOut.write("      local z = math.log(median)\n")
        fileOut.write("      local x = z + sigma * math.sqrt(-2 * math.log(math.random()))\n")
        fileOut.write("      value = math.exp(x) + shift\n")
        fileOut.write("    until value >= min and value <= max\n")
        fileOut.write("    return value\n")
        fileOut.write("  end\n")
        fileOut.write("end\n\n")

        fileOut.write("function generate_radial_distribution(theta_max)\n")
        fileOut.write("  local d = 406.7-24.4\n")
        fileOut.write("  return function()\n")
        fileOut.write("    local angle\n")
        fileOut.write("    local radius_max = d * math.tan(theta_max * math.pi / 180)\n")
        fileOut.write("    repeat\n")
        fileOut.write("      local random_value = math.random() * radius_max\n")
        fileOut.write("      angle = math.acos(random_value/radius_max) * theta_max^2 \n")
        fileOut.write("    until angle >= 0 and angle <= theta_max\n")
        fileOut.write("    return angle\n")
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
        fileOut.write(f"    ke = distribution(lognormal({medianEnergy}, {energySigma}, {shift}, {min_energy}, {max_energy})),\n")
        fileOut.write("    az =  single_value {0},\n")
        fileOut.write(f"    el =  distribution(generate_radial_distribution({max_angle})), \n")
        fileOut.write("    cwf = 1,\n")
        fileOut.write("    color = 0,\n")
        fileOut.write("    position =  sphere_distribution {\n")
        fileOut.write("      center = vector(24.4, 0, 0),\n")
        fileOut.write("      radius = 2,\n")
        fileOut.write("      fill = true")
        fileOut.write("    }\n")
        fileOut.write("  }\n")
        fileOut.write("}")

def fastAdj(voltageArray, potArrayFile):
    # pa0 file directory
    potArrLoc = potArrayFile
    # initialize string that will be supplied as argument of voltage values
    voltArgString = str()
    # convert supplied potential voltages into a string
    for i in range(voltageArray.size):
        # setup values for electrode number and voltage
        electrodeNumber = i + 1
        voltage = voltageArray[i]
        if((i+1) == voltageArray.size):
            stringToAdd = str(electrodeNumber) + "=" + str(voltage)
        else:
            stringToAdd = str(electrodeNumber) + "=" + str(voltage) + ","
        # add current electrode's parameters to string
        voltArgString = voltArgString + stringToAdd

    # go to simion's working directory and call simion
    subprocess.run("simion.exe --nogui fastadj " + potArrLoc + " " + voltArgString)

def fly(sim_cmd, fly_file, out_file, rec_file, bench):
    '''
    Fly n_parts particles using the particle probability distributions defined in self.parts.
    Parallelizes the fly processes by spawing a number of instances associated with the
    number of cores of the processing computer. Resulting particle data is stored in
    self.data as a simPyon.data.sim_data object.

    Parameters
    ----------
    n_parts: int
        number of particles to be flown. n_parts/cores number of particles is flown in each
        instance of simion on each core.
    cores: int
        number of cores to use to process the fly particles request. Default initializes
        a simion instance to run on each core.
    surpress_output: bool
        With surpress_output == True, the preparation statement from one of the simion
        instances is printed to cmd.
    '''

    start_time = time.time()

    # Fly the particles in parallel and scrape the resulting data from the shell
    loc_com = r"fly  "
    loc_com += r" --recording-output=" + out_file
    loc_com += r" --recording=" + rec_file
    loc_com += r" --particles=" + fly_file
    loc_com += r" -- restore-potentials=0 " + bench
    commands = loc_com
    subprocess.run(sim_cmd + ' ' + commands)
    print(time.time() - start_time)

# helper method that makes the call to run simion to fly particles.  This method links up all the required file
# directories into a single call to simion, which will then fly the simulation and create an outputFile
# fly2FileDir contains the definition of the particles that should be simulated.
# This file is made with the method 'makeFLY2File()'
# outputFile is the directory to which the output log will be saved.  This log is a summary of the simulation
# results, and is made by simion as simion runs the simulation
# recordingFile is a special file that has the recording options.  This is a file that can only be made in simion -
# it is a binary file that has a bunch of flags to tell the program what to record during simulation.
# the iobFileLoc is directory to the .IOB file, which is the ion bench file.  I am not fully sure that I understand what
# this is, but I think it is a file that links the potential arrays to the simulation.
def runSimion(fly2File, voltage_array, outputFile, recordingFile, iobFileLoc, potArrLoc, baseDir):
    # Check if outputFile exists and delete it if it does
    if os.path.isfile(outputFile):
        os.remove(outputFile)
    print("flying!")

    # Change to the SIMION working directory
    original_cwd = os.getcwd()
    print(original_cwd)
    os.chdir(r"C:\Users\proxi\Downloads\Simion_8-1-20230825T223627Z-001\Simion_8-1")

    # Convert voltage_array to a space-separated string
    #voltage_string = ' '.join(map(str, voltage_array))
    fastAdj(voltage_array, potArrLoc)

    flyCommand = f"fly --recording-output=\"{outputFile}\" --recording=\"{recordingFile}\" " \
                 f"--particles=\"{fly2File}\" --restore-potentials=0 \"{iobFileLoc}\""

    # Construct the full command
    fullCommand = f"simion.exe --nogui {flyCommand}"
    # Go back to the base directory and delete temporary files
    print("starting flym")
    subprocess.run(fullCommand, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("finished flym")
    os.chdir(baseDir)
    os.system('del *.tmp')
    os.chdir(original_cwd)


if __name__ == '__main__':
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{string} is not a valid path")

    ltspice_path = "C:\\Users\\proxi\\AppData\\Local\\Programs\\ADI\\LTspice\\LTspice.exe"
    baseDir = "C:/Users/proxi/Documents/coding/simulations/TOF_simulation"
    iobFileLoc = baseDir + "/TOF_simulation.iob"
    recordingFile = baseDir + "/TOF_simulation.rec"
    potArrLoc = baseDir + "/copiedArray.PA0"
    lua_path = baseDir + "/TOF_simulation.lua"
    current_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='code for running LTSpice simulations')

    # Add arguments
    parser.add_argument('retardation', metavar='N', type=int, help='Required retardation value')
    parser.add_argument('simulation_dir', type=dir_path, help='Required simulation directory')
    parser.add_argument('--front_voltage', type=float, help='Optional front voltage value')
    parser.add_argument('--back_voltage', type=float, help='Optional back voltage value')

    args = parser.parse_args()
    print(args.front_voltage, args.back_voltage)

    Fly2File = args.simulation_dir + "\\TOF_simulation\\TOF_simulation.fly2"
    simion_output_path = args.simulation_dir + \
                         f"\\TOF_simulation\\simion_output\\test_R{args.retardation}.txt"

    cir_filepath = current_dir + "\\voltage_divider.cir"
    raw_file_path = args.simulation_dir + f"\\ltspice\\voltage_divider_R{args.retardation}.raw"

    ltspice_out_dir = args.simulation_dir + "\\ltspice"

    # Modify the .cir file with new voltages
    generate_fly2File_lognorm(Fly2File, args.retardation, args.retardation+1200, numParticles=1000,
                              medianEnergy=np.exp(2), energySigma=2, shift=args.retardation-10, max_angle=3)
    #generate_fly2File(Fly2File, numParticles=5000, minEnergy=args.retardation-100,
    #                  maxEnergy=args.retardation+1200)
    new_voltages, resistor_values = calculateVoltage_NelderMeade(args.retardation)
    #print(new_voltages)
    #new_cir_filepath = modify_cir_file(cir_filepath, new_voltages,
    #                                   args.simulation_dir + "\\ltspice", args.retardation)

    # Run the LTspice simulation
    #run_simulation(ltspice_path, new_cir_filepath, ltspice_out_dir)

    # Read the .raw file and check current values
    #ok, max_val = check_currents(raw_file_path)
    ok = True
    if ok:
        start = time.time()
        runSimion(Fly2File, new_voltages, simion_output_path, recordingFile, iobFileLoc, potArrLoc, baseDir)
        print(f"simion took: {time.time() - start}")
        parse_and_process_data(simion_output_path)
