import subprocess
import re
from PyLTSpice import RawRead
import os
import sys
from voltage_generator import *


# Function to modify the .cir file with new voltage values
def modify_cir_file(cir_file_path, new_voltages):
    with open(cir_file_path, 'r') as file:
        lines = file.readlines()

    # Regex pattern to find voltage source lines
    pattern = re.compile(r'V\d+\s+n\d+\s+\d+\s+.*')

    # Replace the voltage values in the file
    for i, line in enumerate(lines):
        if pattern.match(line):
            source_num = line.split()[0][1:]
            lines[i] = f"V{source_num} n{source_num} 0 {new_voltages[int(source_num) - 1]}\n"

    # Write the modified lines back to the file
    with open(cir_file_path, 'w') as file:
        file.writelines(lines)


# Function to run the LTspice simulation
def run_simulation(ltspice_path, cir_file_path):
    subprocess.run([ltspice_path, "-b", "-Run", cir_file_path])


# Function to read the .raw file and check current values
def check_currents(raw_file_path):
    def extract_largest_number(lst):
        largest_number = float('-inf')  # Initialize with negative infinity

        for item in lst:
            if item.startswith("I(R"):
                # Extract the numeric part after "I(R"
                try:
                    number = int(item[3:-1])  # Assuming the format is "I(RX)"
                    largest_number = max(largest_number, number)
                except ValueError:
                    # Handle cases where the numeric part is not a valid integer
                    pass

        return largest_number
    ltr = RawRead(raw_file_path)
    names = ltr.get_trace_names()
    # Assume that the currents are labeled as I(R1), I(R2), etc.
    n = extract_largest_number(names)
    for i in range(1, n+1):  # dividing by 2 assumes half are voltage traces
        current_trace = ltr.get_trace(f"I(R{i})")
        if current_trace is None:
            continue
        current_values = current_trace.get_wave(0)  # 0 for DC operating point analysis
        if any(abs(val) > 0.01 for val in current_values):
            return False
        else:
            return True


# Example usage
if __name__ == "__main__":
    # Check if the retardation value is passed as a command-line argument
    if len(sys.argv) > 1:
        retardationValue = float(sys.argv[1])
    else:
        # If not provided, set a default value or exit the script
        print("Please provide a retardation value as a command-line argument.")
        sys.exit(1)


    ltspice_path = "C:\\Users\\proxi\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\LTspice.exe"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # change this slash for linux
    cir_file_path = dir_path + "\\voltage_divider.cir"
    raw_file_path = dir_path + "\\voltage_divider.raw"

    # Modify the .cir file with new voltages
    #new_voltages, resistor_values = calculateVoltage_NelderMeade(retardationValue)
    #modify_cir_file(cir_file_path, new_voltages)

    # Run the LTspice simulation
    #run_simulation(ltspice_path, cir_file_path)

    # Read the .raw file and check current values
    ok = check_currents(raw_file_path)

    print(ok)
