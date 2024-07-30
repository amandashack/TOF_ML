import subprocess
import re
import h5py
from PyLTSpice import RawRead
import os
import argparse
import numpy as np
import sys
sys.path.insert(0, os.path.abspath('..'))
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


def modify_net_file(current_net_filepath, new_voltages, new_filepath):
    # Read the current netlist file
    with open(current_net_filepath, 'r') as file:
        lines = file.readlines()

    # Map to find the voltage lines that need to be updated
    voltage_map = {"V22": -new_voltages[22], "V25": -new_voltages[25], "V1": -new_voltages[27]}

    # Iterate over lines and replace voltages where necessary
    with open(new_filepath, 'w') as new_file:
        for line in lines:
            for key in voltage_map:
                if f'"{key}"' in line:
                    parts = line.split()
                    new_value = f'"{voltage_map[key]:.6f}"'
                    parts[1] = new_value  # Replace the old voltage value
                    line = ' '.join(parts) + '\n'
            new_file.write(line)


# Function to run the LTspice simulation
def run_spice_simulation(spice_path, net_file_path):
    try:
        subprocess.run([spice_path, "-b", "-Run", net_file_path],
                       cwd=os.path.dirname(net_file_path), check=True)
    except FileNotFoundError:
        print(f"LTspice executable not found at {spice_path}. Please check the path.")
    except subprocess.CalledProcessError:
        print("LTspice simulation failed. Please check the input files and configuration.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


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
    if abs(max_val) > 0.001:
        return 0, max_val
    else:
        return 1, max_val


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='code for running LTSpice simulations')
    parser.add_argument('ltspice_dir', type=str, help='Required output directory for '
                                                      'LTspice simulation files')
    parser.add_argument('retardation', type=int, help='Required retardation value')
    parser.add_argument('--blade22', type=float, default=0.11248,
                        help='Optional blade 22 ratio value, default is 0.11248')
    parser.add_argument('--blade25', type=float, default=0.1354,
                        help='Optional blade 25 ratio value, default is 0.1354')

    args = parser.parse_args()

    # Validate output directory
    if not os.path.isdir(args.ltspice_dir):
        raise ValueError("The specified LTspice directory does not exist. Please provide a valid directory.")

    # Validate numerical inputs (example ranges are hypothetical)
    if args.retardation <= 0:
        raise ValueError("Retardation must be a positive integer.")

    if args.blade22 > 1 or args.blade25 > 1:
        raise ValueError("Voltage ratio must be less than 1.")

    if args.blade22 < -1 or args.blade25 < -1:
        raise ValueError("Voltage ratio must be greater than -1.")

    ltspice_path = "C:\\Users\\proxi\\AppData\\Local\\Programs\\ADI\\LTspice\\LTspice.exe"
    base_dir = args.ltspice_dir
    base_filename = f"spice_{args.retardation}_{args.blade22}_{args.blade25}.raw"

    new_net_filepath = os.path.join(base_dir, base_filename.replace('.raw', '.net'))
    raw_file_path = os.path.join(base_dir, base_filename)

    net_filepath = os.path.join(base_dir, "MRCO_NM_Base_LTspice2.net")

    # Modify the .cir file with new voltages
    new_voltages, resistor_values = calculateVoltage_NelderMeade(-args.retardation,
                                                                 mid1_ratio=args.blade22,
                                                                 mid2_ratio=args.blade25)
    modify_cir_file(net_filepath, new_voltages, new_net_filepath)

    # Run the LTspice simulation
    run_spice_simulation(ltspice_path, new_net_filepath)
    print(new_voltages)

    # Read the .raw file and check current values
    ok, max_val = check_currents(raw_file_path)
    print(ok, max_val)

    """with h5py.File(h5_filename, 'a') as h5file:
        # Create a nested group path based on retardation, front_voltage, and back_voltage
        group_path = f"{args.retardation}/{args.front_voltage}/{args.back_voltage}"
        # Navigate through the hierarchy, creating missing groups along the path
        current_group = h5file
        for subpath in group_path.split('/'):
            if subpath not in current_group:
                current_group = current_group.create_group(subpath)
            else:
                current_group = current_group[subpath]

        # Create datasets within the deepest subgroup
        current_group.create_dataset('voltages', data=new_voltages)
        current_group.create_dataset('check', data=np.array([ok, max_val]))

        # Generate and save a unique binary encoding for the combination
        encoding = create_binary_encoding(
            [args.retardation, int(args.front_voltage * 1e5), int(args.back_voltage * 1e5)])
        current_group.attrs['binary_encoding'] = encoding

        print(f"Results saved to {h5_filename} under {group_path} with encoding {encoding}")"""

