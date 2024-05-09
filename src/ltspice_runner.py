import subprocess
import re

from PyLTSpice import RawRead
import os
import argparse
from voltage_generator import *


# Function to modify the .cir file with new voltage values
def modify_cir_file(cir_file_path, new_voltages, new_filepath):
    #base_name = os.path.basename(cir_file_path)
    #name, ext = os.path.splitext(base_name)
    #new_filename = f"{name}_R{retardation}{ext}"
    #new_file_path = os.path.join(output_dir, new_filename)
    voltages = [-new_voltages[22], -new_voltages[25], -new_voltages[27]]
    # Open the file and read the lines
    with open(cir_file_path, 'r') as file:
        lines = file.readlines()

    # Regex pattern to find voltage source lines specifically for V22, V25, and V1
    pattern = re.compile(r'(V22|V25|V1)\s+0\s+N\d+\s+.*')

    # Mapping voltage source names to new_voltages indices
    voltage_mapping = {'V22': 0, 'V25': 1, 'V1': 2}

    # Replace the voltage values in the file
    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            source_name = match.group(1)
            # Extract the node number from the line
            node_num = line.split()[2]
            # Replace the line with the new voltage value
            lines[i] = f"{source_name} 0 {node_num} {voltages[voltage_mapping[source_name]]}\n"

    # Write the modified lines to the new file in the different directory
    with open(new_filepath, 'w') as file:
        file.writelines(lines)

    return new_filepath  # Return the path to the new file for further processing


# Function to run the LTspice simulation
def run_simulation(spice_path, cir_file_path):
    # Save the current directory
    original_cwd = os.getcwd()

    # Change the current directory to the desired output directory
    os.chdir(os.path.dirname(cir_file_path))

    try:
        subprocess.run([spice_path, "-b", "-Run", cir_file_path])
    except FileNotFoundError as e:
        print(f"An error occurred: {e}")
    finally:
        # Change back to the original directory
        os.chdir(original_cwd)


# Function to read the .raw file and check current values
def check_currents(fp):
    def extract_resistor_current_names(lst):
        cn = []
        for item in lst:
            if item.startswith("I(R"):
                cn.append(item)
        return cn

    ltr = RawRead(fp)
    names = ltr.get_trace_names()
    # Assume that the currents are labeled as I(R1), I(R2), etc.
    resistor_current_names = extract_resistor_current_names(names)

    # Check each current trace
    max_val = float(-np.Inf)
    for name in resistor_current_names:
        current_trace = ltr.get_trace(name)
        if current_trace is None:
            continue
        current_value = abs(current_trace.data[-1])  # 0 for DC operating point analysis

        if current_value > max_val: max_val = current_value
    if abs(max_val) > 0.01:
        return 0, max_val
    else:
        return 1, max_val


# Example usage
if __name__ == "__main__":
    # Check if the retardation value is passed as a command-line argument
    parser = argparse.ArgumentParser(description='code for running LTSpice simulations')

    # Add arguments
    parser.add_argument('retardation', type=int, help='Required retardation value')
    parser.add_argument('ltspice_dir', type=str, help='Required output directory including the filename')
    parser.add_argument('--front_voltage', type=float, help='Optional front voltage value')
    parser.add_argument('--back_voltage', type=float, help='Optional back voltage value')
    parser.add_argument('--nose_cone', type=float, help='Optional nose cone value')

    args = parser.parse_args()

    ltspice_path = "C:\\Users\\proxi\\AppData\\Local\\Programs\\ADI\\LTspice\\LTspice.exe"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    cir_filepath = dir_path + "\\voltage_divider.cir"
    base_dir = os.path.dirname(args.ltspice_dir)
    base_filename = os.path.basename(args.ltspice_dir)

    # Create .cir and .raw file paths
    new_cir_filepath = os.path.join(base_dir, base_filename.replace('.raw', '.cir'))
    raw_file_path = args.ltspice_dir

    # Modify the .cir file with new voltages
    new_voltages, resistor_values = calculateVoltage_NelderMeade(args.retardation,
                                                                 args.front_voltage,
                                                                 args.back_voltage,
                                                                 args.nose_cone)
    modify_cir_file(cir_filepath, new_voltages, new_cir_filepath)

    # Run the LTspice simulation
    run_simulation(ltspice_path, new_cir_filepath)

    # Read the .raw file and check current values
    ok, max_val = check_currents(raw_file_path)

    print(max_val, ok)
