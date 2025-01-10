import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib
import tempfile
import shutil
from multiprocessing import Pool, cpu_count
from collections import defaultdict
matplotlib.rcParams['pdf.fonttype'] = 42
sys.path.insert(0, os.path.abspath('../..'))
from tof_ml.simulations.voltage_generator import calculateVoltage_NelderMeade
from run_simion import parse_and_process_data, runSimion
from loaders.load_and_save import read_h5_data


ResultsDir = r"C:\Users\proxi\Documents\coding\TOF_data\discrete_test"
SimionDir = r"C:\Users\proxi\Documents\coding\TOF_ML_backup\simulations\TOF_simulation"
iobFileLoc = SimionDir + "/TOF_simulation.iob"
recordingFile = SimionDir + "/TOF_simulation.rec"
potArrLoc = SimionDir + "/copiedArray.PA0"
lua_path = SimionDir + "/TOF_simulation.lua"
Fly2File = SimionDir + "/TOF_simulation.fly2"


def generate_fly2File_logarithmic(filenameToWriteTo, min_energy, max_energy,
                                  num_particles=20000, num_log_points=100, max_angle=3):
    # Generate 100 logarithmically spaced points between min_energy and max_energy
    log_points = np.logspace(np.log10(min_energy), np.log10(max_energy), num=num_log_points)

    # Select 25,000 data points from the logarithmic distribution
    sampled_points = np.random.choice(log_points, size=num_particles, replace=True)

    # Check if .fly2 file with this name already exists
    fileExists = os.path.isfile(filenameToWriteTo)

    # Delete previous copy, if there is one
    if fileExists:
        os.remove(filenameToWriteTo)

    # Open up file to write to
    with open(filenameToWriteTo, "w") as fileOut:
        # Write the Lua function for the custom logarithmic distribution at the beginning of the file
        fileOut.write("-- Define custom logarithmic distribution function\n")
        fileOut.write("local math = require('math')\n")
        fileOut.write("local distribution_points = {\n")
        for point in sampled_points:
            fileOut.write(f"  {point},\n")
        fileOut.write("}\n")
        fileOut.write("function sample_from_distribution()\n")
        fileOut.write("  return distribution_points[math.random(1, #distribution_points)]\n")
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

        # Now define the particle distribution using the custom logarithmic distribution
        fileOut.write("particles {\n")
        fileOut.write("  coordinates = 0,\n")
        fileOut.write("  standard_beam {\n")
        fileOut.write(f"    n = {num_particles},\n")
        fileOut.write("    tob = 0,\n")
        fileOut.write("    mass = 0.000548579903,\n")
        fileOut.write("    charge = -1,\n")
        fileOut.write("    ke = distribution(sample_from_distribution),\n")
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



def run_simulation(args):
    num_points, retardation, mid1_ratio, mid2_ratio, baseDir = args

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
        generate_fly2File_logarithmic(Fly2File, 0.3, 1000,
                                      num_log_points=num_points, max_angle=5)

        output_dir = os.path.join(baseDir, f"R{np.abs(retardation)}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        simion_output_path = os.path.join(output_dir, f"NM_R1_{num_points}.txt")

        runSimion(Fly2File, new_voltages, simion_output_path, recordingFile, iobFileLoc, potArrLoc, temp_dir)
        parse_and_process_data(simion_output_path)

        return simion_output_path
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

def parallel_pool(num_cores, simulation_args):
    with Pool(processes=num_cores) as pool:
        results = list(pool.imap_unordered(run_simulation, simulation_args))
    return results


def plot_tof_vs_ke_with_derivative(data, output_dir, file_name):
    ke = data['initial_ke']
    tof = data['tof']
    x_tof = data['x']

    # Mask the data where x_tof is greater than 403.6
    mask = x_tof > 403.6
    ke = ke[mask]
    tof = tof[mask]

    # Group by kinetic energy and calculate the mean TOF for each KE
    ke_to_tof = defaultdict(list)
    for k, t in zip(ke, tof):
        ke_to_tof[k].append(t)
    ke_unique = np.array(list(ke_to_tof.keys()))
    tof_mean = np.array([np.mean(ke_to_tof[k]) for k in ke_unique])

    # Sort by kinetic energy
    sorted_indices = np.argsort(ke_unique)
    ke_sorted = np.log2(ke_unique[sorted_indices])
    tof_sorted = np.log2(tof_mean[sorted_indices])

    # Calculate the derivative of TOF with respect to KE
    derivative = np.gradient(tof_sorted, ke_sorted)

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots()

    # Scatter plot TOF vs. KE
    color = 'tab:blue'
    ax1.set_xlabel('Kinetic Energy (eV)')
    ax1.set_ylabel('Time of Flight (µs)', color=color)
    ax1.scatter(ke_sorted, tof_sorted, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a secondary y-axis for the derivative plot
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Derivative of TOF with respect to KE', color=color)
    ax2.plot(ke_sorted, derivative, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Adjust layout to prevent overlap and save the plot
    plt.title("TOF vs. KE and its Derivative")
    fig.tight_layout()
    output_file = os.path.join(output_dir, f'tof_vs_ke_with_derivative_{file_name}.png')
    plt.savefig(output_file)
    plt.close()

    print(f"Plot saved to: {output_file}")


def process_simulation_results(results_dir):
    # Iterate through each H5 file in the results directory
    results_dir = os.path.join(results_dir, 'R1')
    for file_name in os.listdir(results_dir):
        if file_name.endswith('.h5'):
            file_path = os.path.join(results_dir, file_name)
            print(f"Processing file: {file_path}")

            # Read data from H5 file
            data = read_h5_data(file_path)

            # Plot TOF vs. KE with the derivative
            plot_tof_vs_ke_with_derivative(data, results_dir, os.path.splitext(file_name)[0])


def display_points_hist(min_energy, max_energy, n_points, n_samples):
    # Step 1: Generate 100 points logarithmically spaced between 0.1 and 1000
    log_points = np.logspace(np.log2(min_energy), np.log2(max_energy), num=n_points, base=2)

    # Step 2: Randomly sample 25,000 data points from the 100 discrete points
    sampled_points = np.random.choice(log_points, size=n_samples, replace=True)
    # Optional: Plot the distribution of sampled points to verify
    plt.hist(sampled_points, bins=n_points, log=False)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sampled Points from Logarithmic Distribution')
    plt.show()


if __name__ == '__main__':
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{string} is not a valid path")

    parser = argparse.ArgumentParser(
        description='code for running Simion simulations for collection efficiency analysis')

    parser.add_argument(
        "--num_points",
        nargs="*",
        type=int,
        default=[30, 100, 1000, 5000],
    )

    parser.add_argument(
        "--retardation",
        nargs="*",
        type=int,
        default=[1],
    )

    parser.add_argument(
        "--mid1_ratio",
        nargs="*",
        type=float,
        default=[0.11248],
    )

    parser.add_argument(
        "--mid2_ratio",
        nargs="*",
        type=float,
        default=[0.1354],
    )

    args = parser.parse_args()

    # Prepare list of all combinations
    simulation_args = [(num_points, retardation, mid1_ratio, mid2_ratio, ResultsDir)
                       for num_points in args.num_points
                       for retardation in args.retardation
                       for mid1_ratio in args.mid1_ratio
                       for mid2_ratio in args.mid2_ratio]

    # Determine the number of CPU cores to use
    num_cores = cpu_count()  # Specify the number of cores to use

    # Record execution time for parallel execution
    #parallel_times = parallel_pool(num_cores, simulation_args)
    process_simulation_results(ResultsDir)


