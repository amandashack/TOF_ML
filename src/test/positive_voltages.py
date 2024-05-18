import argparse
import os
import numpy as np
import sys
sys.path.insert(0, os.path.abspath('..'))
from voltage_generator import calculateVoltage_NelderMeade
from run_simion import parse_and_process_data, runSimion, generate_fly2File_lognorm

def generate_fly2File2(filenameToWriteTo, max_energy, numParticles=100, max_angle=3):

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
        fileOut.write("    ke =  uniform_distribution {\n")
        fileOut.write("      min = " + str(max_energy) + ",\n")
        fileOut.write("      max = " + str(max_energy) + "\n")
        fileOut.write("    },\n")
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


if __name__ == '__main__':
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{string} is not a valid path")

    baseDir = "C:/Users/proxi/Documents/coding/TOF_ML/simulations/TOF_simulation"
    iobFileLoc = baseDir + "/TOF_simulation.iob"
    recordingFile = baseDir + "/TOF_simulation.rec"
    potArrLoc = baseDir + "/copiedArray.PA0"
    lua_path = baseDir + "/TOF_simulation.lua"
    Fly2File = baseDir + "/TOF_simulation.fly2"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    retardation = 0

    parser = argparse.ArgumentParser(description='code for running LTSpice simulations')

    # Add arguments
    # turn this into a list of values to get through
    parser.add_argument(
        "--front_voltage",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # default if nothing is provided
    )
    parser.add_argument(
        "--kinetic_energy",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=float,
        default=[0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # default if nothing is provided
    )

    args = parser.parse_args()
    for front_voltage in args.front_voltage:
        print(front_voltage)
        for ke in args.kinetic_energy:
            print(ke)
            if ke == 0.1:
                simion_output_path = (f"C:\\Users\\proxi\\Documents\\coding\\TOF_ML\\simulations\\"
                                      f"TOF_simulation\\simion_output\\positive_voltage\\nonNM\\"
                                      f"test_R{retardation}_{front_voltage}_0.txt")
            else:
                simion_output_path = (f"C:\\Users\\proxi\\Documents\\coding\\TOF_ML\\simulations\\"
                                      f"TOF_simulation\\simion_output\\positive_voltage\\nonNM\\"
                                      f"test_R{retardation}_{front_voltage}_{int(ke)}.txt")

            generate_fly2File2(Fly2File, float(ke), numParticles=1000, max_angle=2.5)
            #generate_fly2File_lognorm(Fly2File, 0.1,  20, numParticles=100,
            #                          medianEnergy=np.exp(2), energySigma=2, shift=-10, max_angle=3)
            new_voltages, resistor_values = calculateVoltage_NelderMeade(0, voltage_front=0)
            new_voltages[0] = front_voltage
            print(new_voltages)
            runSimion(Fly2File, new_voltages, simion_output_path, recordingFile, iobFileLoc, potArrLoc, baseDir)
            parse_and_process_data(simion_output_path)
