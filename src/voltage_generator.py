import sys
import numpy as np
from control_methods import voltageArrayGeneratorWrapperOneOverR

def calculateNMVoltageBridge(voltageFront, voltageBack, midOneVoltage, midTwoVoltage):
    # denote the locations along the voltage node of the mid voltage sources.
    # They are counted assuming the first value is numbered "0"
    midOneLeadLocation = 22
    midTwoLeadLocation = 25
    frontLocation = 0
    # resistor values (in MegaOhms) for NM as selected by Ave
    resistorValues = np.array([3.08, 2.43, 1.75, 1.25, 0.897, 0.664, 0.528, 0.446, 0.457, 0.483,
                               0.528, 0.577, 0.621, 0.650, 0.657, 0.639, 0.593, 0.520, 0.422,
                               0.305, 0.178, 0.048, 0.072, 0.167, 0.220, 2.13])

    # setup an array that will hold the relative voltage values for each blade/lead.
    # This will be done by building up a resistor bridge
    numResistors = len(resistorValues)
    numVoltages = numResistors + 1 # the number of voltages is the number of resistors plus one,
    # since I want voltages at leads , which are bridged by resistors.
    voltages = np.zeros(numVoltages)
    # to calculate NM voltages, need to treat it as a series of voltage divider bridges,
    # depending on where the leads are located.
    #take care of first monotonic series, between front mesh and midOne
    # perform setup
    voltageValuesRelativeNow = np.zeros(midOneLeadLocation + 1 - frontLocation)
    resistanceSum = 0
    # handle case of voltage lead prior to any resistors being encountered
    voltageValuesRelativeNow[0] = resistanceSum
    # run through the loop of having the resistors
    for i in range(frontLocation, midOneLeadLocation):
        resistanceSum += resistorValues[i]
        voltageValuesRelativeNow[i + 1 - frontLocation] = resistanceSum
    # completed the loop.  normalize by total resistance crossed:
    voltageValuesRelativeNow = voltageValuesRelativeNow/resistanceSum
    # change normalized offset to represent real voltages, depending on those supplied
    voltageDifference = (midOneVoltage - voltageFront)
    voltages[frontLocation:(midOneLeadLocation + 1)] = voltageDifference*voltageValuesRelativeNow + voltageFront
    # take care of second monotonic series, between midOne and midTwo
    # perform setup
    voltageValuesRelativeNow = np.zeros(midTwoLeadLocation + 1 - midOneLeadLocation)
    resistanceSum = 0
    # handle case of voltage lead prior to any resistors being encountered
    voltageValuesRelativeNow[0] = resistanceSum
    # run through the loop of having the resistors
    for i in range(midOneLeadLocation, midTwoLeadLocation):
        resistanceSum += resistorValues[i]
        voltageValuesRelativeNow[i + 1 - midOneLeadLocation] = resistanceSum
    # completed the loop.  normalize by total resistance crossed:
    voltageValuesRelativeNow = voltageValuesRelativeNow/resistanceSum
    # change normalized offset to represent real voltages, depending on those supplied
    voltageDifference = (midTwoVoltage - midOneVoltage)
    voltages[midOneLeadLocation:(midTwoLeadLocation + 1)] = voltageDifference*voltageValuesRelativeNow + midOneVoltage
    backLocation = numResistors
    # take care of final monotonic series, between midTwo and back mesh
    # perform setup
    voltageValuesRelativeNow = np.zeros(backLocation + 1 - midTwoLeadLocation)
    resistanceSum = 0
    # handle case of voltage lead prior to any resistors being encountered
    voltageValuesRelativeNow[0] = resistanceSum
    # run through the loop of having the resistors
    for i in range(midTwoLeadLocation, backLocation):
        resistanceSum += resistorValues[i]
        voltageValuesRelativeNow[i + 1 - midTwoLeadLocation] = resistanceSum
    # completed the loop.  normalize by total resistance crossed:
    voltageValuesRelativeNow = voltageValuesRelativeNow/resistanceSum
    # change normalized offset to represent real voltages, depending on those supplied
    voltageDifference = (voltageBack - midTwoVoltage)
    voltages[midTwoLeadLocation:(backLocation + 1)] = voltageDifference*voltageValuesRelativeNow + midTwoVoltage
    return voltages, resistorValues


def voltageArrayGeneratorWrapperNM(voltageFront, voltageBack, midOneVoltage, midTwoVoltage):
    # current (10/9/2020) version of file has 37 electrodes
    voltageArray = np.zeros(37)

    # call a helper method to get voltage settings of NM lens stack, given a front and back mesh voltage
    lensVoltages, resistor_values = calculateNMVoltageBridge(voltageFront, voltageBack, midOneVoltage, midTwoVoltage)

    voltageArray[1:26] = lensVoltages[1:26]#control lens stack voltages
    voltageArray[27] = lensVoltages[26] #set back mesh voltage to be same as final electric lens voltage
    voltageArray[26] = lensVoltages[0]  #set front mesh voltage to be same as first lens voltage
    voltageArray[[29, 30, 34, 36]] = lensVoltages[0] #set potential for nose and surrounding material
    voltageArray[0] = lensVoltages[0] #set the voltage of outer construct of model - material that should be on outside of flight path
    voltageArray[28] = voltageArray[27] #control voltage of MCP mesh to be same as voltage of back mesh
    voltageArray[31:33] = voltageArray[28] #control inner surface of flight tube to be same voltage as MCP mesh
    voltageArray[33] = voltageArray[28] + 300 #set MCP front voltage to be 300 V above the MCP mesh
    voltageArray[35] = lensVoltages[0] #control voltage of mesh near electron generation point, to define voltage
    # near area where electrons are created
    return voltageArray, resistor_values


def calculateVoltage_NelderMeade(retardationValue, voltageMidOne=None, voltageMidTwo=None, voltageFront=None):
    # setup fast adjust voltages
    if not voltageFont:
        voltageFront = 0
    voltageBack = -1 * abs(retardationValue)  # only talking about electrons
    if not voltageMidOne:
        voltageMidOne = voltageBack + 0.11248 * (voltageFront - voltageBack)
    if not voltageMidTwo:
        voltageMidTwo = voltageBack + 0.1354 * (voltageFront - voltageBack)
    # #run for NM
    voltageArray, resistor_values = voltageArrayGeneratorWrapperNM(voltageFront, voltageBack, voltageMidOne, voltageMidTwo)
    return voltageArray, resistor_values


def calculateVoltage_OneoverR(retardationValue):
    # setup fast adjust voltages
    voltageFront = 0
    voltageBack = -1 * retardationValue

    # #run for NM
    voltageArray, resistor_values = voltageArrayGeneratorWrapperOneOverR(voltageFront, voltageBack)

    return voltageArray, resistor_values


def generate_netlist(voltage_array, resistor_values, filename="voltage_divider.cir"):
    # Begin netlist with comments for readability
    netlist = "* Auto-generated netlist for an LTspice simulation\n\n"

    # Add resistor R1, which starts from node 0
    netlist += "R1 0 N003 {0}Meg\n".format(resistor_values[0])

    # Add resistors R2 to R21 in the chain from N003 to N024
    for i, resistor_value in enumerate(resistor_values[1:21], start=2):  # resistors R2 to R21
        netlist += "R{0} N{1:03d} N{2:03d} {3}Meg\n".format(i, i + 1, i + 2, resistor_value)

    # Add resistors R22, R23, and R25 which are connected to N001 and N002
    netlist += "R22 N023 N001 {0}Meg\n".format(resistor_values[21])
    netlist += "R23 N001 N024 {0}Meg\n".format(resistor_values[22])
    netlist += "R25 N025 N002 {0}Meg\n".format(resistor_values[24])

    # Add resistor R26 which connects N026 to N002
    netlist += "R26 N026 N002 {0}Meg\n".format(resistor_values[25])

    # Add voltage sources V22 and V25
    netlist += "V22 0 N001 {0}\n".format(-voltage_array[22])
    netlist += "V25 0 N002 {0}\n".format(-voltage_array[25])

    # Add voltage source V1 at node N026
    netlist += "V1 0 N026 {0}\n".format(-voltage_array[27])

    # Add simulation command
    netlist += "\n.op\n"

    # End the netlist
    netlist += ".end\n"

    # Write the netlist to a file
    with open(filename, "w") as file:
        file.write(netlist)

    return netlist


def create_ltspice_asc_file(voltage_array, resistor_values, filename="voltage_divider.asc"):
    # Define the header for the LTspice .asc file
    asc_content = ["Version 4", "SHEET 1 880 680", "WIRE WireLine"]
    # Initialize component positions
    x_origin, y_origin = 0, 0
    delta_x, delta_y = 160, 0

    # Add the voltage source and ground
    asc_content.append(f"SYMBOL voltage {x_origin} {y_origin} R0")
    asc_content.append(f"SYMATTR InstName V1")
    asc_content.append(f"SYMATTR Value {voltage_array[0]}")
    asc_content.append(f"TEXT {x_origin - 60} {y_origin + 60} Left 2 !.tran 1")

    # Add the resistors and their values
    for i, (resistor_value, voltage) in enumerate(zip(resistor_values, voltage_array[1:]), start=1):
        x_current = x_origin + i * delta_x
        asc_content.append(f"SYMBOL res {x_current} {y_origin} R0")
        asc_content.append(f"SYMATTR InstName R{i}")
        asc_content.append(f"SYMATTR Value {resistor_value}Meg")
        # Add a wire between the resistors
        if i > 1:
            x_previous = x_origin + (i - 1) * delta_x
            asc_content.append(f"WIRE {x_previous + 48} {y_origin + 64} {x_current} {y_origin + 64}")

    # Add wires to complete the circuit to ground
    asc_content.append(f"WIRE {x_origin} {y_origin + 64} {x_origin} {y_origin}")
    asc_content.append(f"WIRE {x_origin + 48} {y_origin + 64} {x_origin + 48} {y_origin + 64}")
    asc_content.append(f"FLAG {x_origin + 48} {y_origin + 64} 0")
    asc_content.append(f"FLAG {x_origin} {y_origin} 0")

    # Write the .asc file content to a file
    with open(filename, 'w') as file:
        file.write('\n'.join(asc_content))

    return asc_content


if __name__ == "__main__":
    # Check if the retardation value is passed as a command-line argument
    if len(sys.argv) > 1:
        retardationValue = float(sys.argv[1])
    else:
        # If not provided, set a default value or exit the script
        print("Please provide a retardation value as a command-line argument.")
        sys.exit(1)

    # Calculate the voltage and resistor values
    voltage_array, resistor_values = calculateVoltage_NelderMeade(retardationValue)

    # Generate the netlist file
    n = generate_netlist(voltage_array, resistor_values)
    print(n)

