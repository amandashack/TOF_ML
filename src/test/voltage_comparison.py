from PyLTSpice import RawRead
import os
import argparse
import sys
sys.path.insert(0, os.path.abspath('..'))
from voltage_generator import *
from ltspice_runner import *
import matplotlib.pyplot as plt
from plotter import get_cmap


def make_comparison(fp1, fp2, new_voltages):
    def compare_lists(l1, l2, tolerance):
        discrepancies = []
        if len(l1) != len(l2):
            print("Lists are not the same size")
            return
        for index, (a, b) in enumerate(zip(l1, l2)):
            if abs(a - b) > tolerance:
                discrepancies.append((index, a, b))
        if discrepancies:
            print("Discrepancies found at the following indices (index, value in list1, value in list2):")
            for discrepancy in discrepancies:
                print(discrepancy)
            return False
        else:
            print("The lists are equivalent up to the specified tolerance.")
            return True

    def extract_resistor_current_names(lst):
        cn = []
        vn = []
        for item in lst:
            if item.startswith("I(R"):
                cn.append(item)
            elif item.startswith("I(V"):
                cn.append(item)
            elif item.startswith(("V(n0")):
                vn.append(item)
        return cn, vn

    ltr1 = RawRead(fp1)
    names1 = ltr1.get_trace_names()
    ltr2 = RawRead(fp2)
    names2 = ltr2.get_trace_names()
    # Assume that the currents are labeled as I(R1), I(R2), etc.
    current_names1, voltage_names1 = extract_resistor_current_names(names1)
    current_names2, voltage_names2 = extract_resistor_current_names(names2)
    # Check each current trace
    calculated_voltages = new_voltages[1:-11].tolist() + [float(new_voltages[-10])]
    currents1 = []
    voltages1 = []
    for current, voltage in zip(current_names1, voltage_names1):
        current_trace = ltr1.get_trace(current)
        current_value = current_trace.data[-1]
        voltage_trace = ltr1.get_trace(voltage)
        voltage_value = voltage_trace.data[-1]
        currents1.append(current_value)
        voltages1.append(voltage_value)

    currents2 = []
    voltages2 = []
    for current, voltage in zip(current_names2, voltage_names2):
        current_trace = ltr2.get_trace(current)
        current_value = current_trace.data[-1]  # 0 for DC operating point analysis
        voltage_trace = ltr2.get_trace(voltage)
        voltage_value = voltage_trace.data[-1]  # 0 for DC operating point analysis
        currents2.append(current_value)
        voltages2.append(voltage_value)
    print("Compare Amanda to Razib: ", compare_lists(voltages1, voltages2, 0.001))
    print("Compare Amanda to code: ", compare_lists(voltages1, calculated_voltages, 0.001))
    print("Compare Razib to code: ", compare_lists(calculated_voltages, voltages2, 0.001))
    return [voltages1, currents1], [voltages2, currents2]


def plot_relation(ax, ds, x_label, y_label, title=None):
    """
    Plots TOF versus Pass Energy for all or specified retardation values.
    :param ds: data structure you would like to plot, assumed to be a list of dictionaries where
    each list corresponds to a different retardation
    :param specific_retardations: List of retardation values to plot. If None, plots for all retardation values.
    """
    cmap = get_cmap(len(ds.keys()), name='plasma')
    for i, retardation in enumerate(ds.keys()):
        c = cmap(i)
        ax.scatter(ds[retardation]['method1'][0], ds[retardation]['method1'][1], alpha=0.6,
                   label=f"R={retardation}", color=c)
        ax.scatter(ds[retardation]['method2'][0], ds[retardation]['method2'][1], alpha=0.6,
                   marker="^", color=c)
    if title:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    # Check if the retardation value is passed as a command-line argument
    parser = argparse.ArgumentParser(description='code for running LTSpice simulations')

    # Add arguments
    parser.add_argument(
        "--lista",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[0, 100, 300],  # default if nothing is provided
    )
    parser.add_argument('--front_voltage', type=float, help='Optional front voltage value')
    parser.add_argument('--back_voltage', type=float, help='Optional back voltage value')
    parser.add_argument('--nose_cone', type=float, help='Optional nose cone value')

    args = parser.parse_args()

    ltspice_path = "C:\\Users\\proxi\\AppData\\Local\\Programs\\ADI\\LTspice\\LTspice.exe"
    # gets me to the directory below
    base_dir = os.path.dirname(os.path.realpath(__file__))
    dir_path_spice_model = os.path.dirname(base_dir)
    # change this slash for linux
    cir_filepath = dir_path_spice_model + "\\voltage_divider.cir"
    old_cir = "C:/Users/proxi/Downloads/MRCO_NM_Base_LTspice.asc-20231205T203243Z-001/MRCO_NM_Base_LTspice.net"

    sim_results = {}
    for retardation in args.lista:
        # Modify the .cir file with new voltages
        new_voltages, resistor_values = calculateVoltage_NelderMeade(retardation,
                                                                     args.front_voltage,
                                                                     args.back_voltage,
                                                                     args.nose_cone)
        print(new_voltages)
        new_cir_filepath1 = base_dir + f"/lt_spice/amanda_R{retardation}.cir"
        new_cir_filepath2 = base_dir + f"/lt_spice/razib_R{retardation}.cir"
        modify_cir_file(cir_filepath, new_voltages, new_cir_filepath1)
        # razib one goes with his net file
        modify_cir_file(old_cir, new_voltages, new_cir_filepath2)

        # Run the LTspice simulation
        run_simulation(ltspice_path, new_cir_filepath1)
        run_simulation(ltspice_path, new_cir_filepath2)

        base_filename1 = os.path.basename(new_cir_filepath1)
        base_filename2 = os.path.basename(new_cir_filepath2)
        new_raw_filepath1 = os.path.join(base_dir + "\\lt_spice", base_filename1.replace('.cir', '.raw'))
        new_raw_filepath2 = os.path.join(base_dir + "\\lt_spice", base_filename2.replace('.cir', '.raw'))

        sim1, sim2 = make_comparison(new_raw_filepath1, new_raw_filepath2, new_voltages)
        sim_results[retardation] = {'method1': sim1, 'method2': sim2}

    fig, ax = plt.subplots()
    plot_relation(ax, sim_results, "Voltage","Current")
    plt.show()
    # Read the .raw file and check current values
    #ok, max_val = check_currents(raw_file_path)

    #print(max_val, ok)