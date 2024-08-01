import os
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
import sys
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from run_simion import parse_and_process_data, runSimion, generate_fly2File_lognorm
from ltspice_runner import run_spice_simulation, modify_cir_file, check_currents
sys.path.insert(0, os.path.abspath('..'))
from voltage_generator import calculateVoltage_NelderMeade


ResultsDir = r"C:\Users\proxi\Documents\coding\TOF_data"
SimionDir = r"C:\Users\proxi\Documents\coding\TOF_ML_backup\simulations\TOF_simulation"
iobFileLoc = SimionDir + "/TOF_simulation.iob"
recordingFile = SimionDir + "/TOF_simulation.rec"
potArrLoc = SimionDir + "/copiedArray.PA0"
lua_path = SimionDir + "/TOF_simulation.lua"
Fly2File = SimionDir + "/TOF_simulation.fly2"
ltspice_path = r"C:\Users\proxi\AppData\Local\Programs\ADI\LTspice\LTspice.exe"
spice_sim_path = r"C:\Users\proxi\Documents\coding\TOF_ML\simulations\ltspice"



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
        fileOut.write("  local d = 406.7-12.2\n")
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
        fileOut.write("      center = vector(12.2, 0, 0),\n")
        fileOut.write("      radius = 0,\n")
        fileOut.write("      fill = true")
        fileOut.write("    }\n")
        fileOut.write("  }\n")
        fileOut.write("}")


def run_simulation(args):
    retardation, mid1_ratio, mid2_ratio, ke, baseDir = args

    if retardation < 0:
        ke = ke - retardation

    spice_filename = f"spice_{-retardation}_{mid1_ratio}_{mid2_ratio}.raw"

    new_net_filepath = os.path.join(spice_sim_path, spice_filename.replace('.raw', '.net'))
    raw_file_path = os.path.join(spice_sim_path, spice_filename)

    net_filepath = os.path.join(spice_sim_path, "MRCO_NM_Base_LTspice2.net")

    # Modify the .cir file with new voltages
    new_voltages, resistor_values = calculateVoltage_NelderMeade(-retardation,
                                                                 mid1_ratio=mid1_ratio,
                                                                 mid2_ratio=mid2_ratio)
    modify_cir_file(net_filepath, new_voltages, new_net_filepath)

    # Run the LTspice simulation
    run_spice_simulation(ltspice_path, new_net_filepath)

    # Read the .raw file and check current values
    ok, max_val = check_currents(raw_file_path)

    over_current = []

    if ok:
        print("How are you doing? ", ok, )
        # Create a temporary directory for this simulation
        temp_dir = tempfile.mkdtemp()
        try:
            # Copy necessary files to the temporary directory
            iobFileLoc = shutil.copy(os.path.join(SimionDir, "TOF_simulation.iob"), temp_dir)
            recordingFile = shutil.copy(os.path.join(SimionDir, "TOF_simulation.rec"), temp_dir)
            lua_path = shutil.copy(os.path.join(SimionDir, "TOF_simulation.lua"), temp_dir)
            Fly2File = shutil.copy(os.path.join(SimionDir, "TOF_simulation.fly2"), temp_dir)

            # Copy all PA files to the temporary directory
            for pa_file in os.listdir(SimionDir):
                if pa_file.startswith("copiedArray.PA"):
                    shutil.copy(os.path.join(SimionDir, pa_file), temp_dir)

            potArrLoc = os.path.join(temp_dir, "copiedArray.PA0")

            new_voltages, resistor_values = calculateVoltage_NelderMeade(retardation, voltage_front=0,
                                                                         mid1_ratio=mid1_ratio,
                                                                         mid2_ratio=mid2_ratio)
            generate_fly2File2(Fly2File, float(ke), numParticles=1000, max_angle=5)

            if mid1_ratio < 0:
                mid1 = np.abs(mid1_ratio)
                m1sign = "neg"
            else:
                mid1 = mid1_ratio
                m1sign = "pos"
            if mid2_ratio < 0:
                mid2 = np.abs(mid2_ratio)
                m2sign = "neg"
            else:
                mid2 = mid2_ratio
                m2sign = "pos"
            if retardation < 0:
                r = np.abs(retardation)
                rsign = "neg"
            else:
                r = retardation
                rsign = "pos"

            output_dir = os.path.join(baseDir, f"R{np.abs(retardation)}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if ke == 0.1:
                simion_output_path = os.path.join(output_dir, f"sim_{rsign}_R{r}_{m1sign}_{mid1}_{m2sign}_{mid2}_0.txt")
            else:
                simion_output_path = os.path.join(output_dir,
                                                  f"sim_{rsign}_R{r}_{m1sign}_{mid1}_{m2sign}_{mid2}_{float(ke)}.txt")

            runSimion(Fly2File, new_voltages, simion_output_path, recordingFile, iobFileLoc, potArrLoc, temp_dir)
            parse_and_process_data(simion_output_path)

            return simion_output_path
        except Exception as e:
            print(f"Error in simulation: {e}")
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)
    else:
        over_current.append((retardation, mid1_ratio, mid2_ratio))
    return over_current


def check_existing_simulation(args):
    retardation, mid1_ratio, mid2_ratio, ke, baseDir = args

    if mid1_ratio < 0:
        mid1 = np.abs(mid1_ratio)
        m1sign = "neg"
    else:
        mid1 = mid1_ratio
        m1sign = "pos"
    if mid2_ratio < 0:
        mid2 = np.abs(mid2_ratio)
        m2sign = "neg"
    else:
        mid2 = mid2_ratio
        m2sign = "pos"
    if retardation < 0:
        r = np.abs(retardation)
        rsign = "neg"
    else:
        r = retardation
        rsign = "pos"

    # Construct the output path based on the structure provided
    output_dir = os.path.join(baseDir, f"R{np.abs(retardation)}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if ke == 0.1:
        simion_output_path = os.path.join(output_dir, f"sim_{rsign}_R{r}_{m1sign}_{mid1}_{m2sign}_{mid2}_0.h5")
    else:
        simion_output_path = os.path.join(output_dir,
                                          f"sim_{rsign}_R{r}_{m1sign}_{mid1}_{m2sign}_{mid2}_{float(ke)}.h5")

    if os.path.exists(simion_output_path) and os.path.getsize(simion_output_path) >= 89000:
        return None
    else:
        return args


def record_execution_time(num_cores, simulation_args):
    times = []
    start_time = time.time()
    with Pool(processes=num_cores) as pool:
        for i, _ in enumerate(pool.imap_unordered(run_simulation, simulation_args), 1):
            if i % num_cores == 0:
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                print(f"{i} simulations completed in {elapsed_time:.2f} seconds")
    return times


def parallel_pool(num_cores, simulation_args):
    results = []
    with Pool(processes=num_cores) as pool:
        results.append(list(pool.imap_unordered(run_simulation, simulation_args)))
    return results


if __name__ == '__main__':
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{string} is not a valid path")

    parser = argparse.ArgumentParser(
        description='code for running Simion simulations for collection efficiency analysis')

    parser.add_argument(
        "--retardation",
        nargs="*",
        type=int,
        default=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    parser.add_argument(
        "--kinetic_energy",
        nargs="*",
        type=float,
        default=[0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    )

    parser.add_argument(
        "--mid1_ratio",
        nargs="*",
        type=float,
        default=[0.08, 0.11248, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    )

    parser.add_argument(
        "--mid2_ratio",
        nargs="*",
        type=float,
        default=[0.08, 0.1354, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    )

    args = parser.parse_args()



    # Prepare list of all combinations
    simulation_args = [(retardation, mid1_ratio, mid2_ratio, ke, ResultsDir)
                       for retardation in args.retardation
                       for mid1_ratio in args.mid1_ratio
                       for mid2_ratio in args.mid2_ratio
                       for ke in args.kinetic_energy]
    print(len(simulation_args))

    # Filter out existing simulations
    with Pool(processes=cpu_count()) as pool:
        simulation_args = list(filter(None, pool.map(check_existing_simulation, simulation_args)))

    # Determine the number of CPU cores to use
    num_cores = cpu_count()  # Specify the number of cores to use

    # Record execution time for parallel execution
    results = parallel_pool(num_cores, simulation_args)
    print(results)
