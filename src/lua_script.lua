-- Load the workbench
simion.workbench_program()

-- Function to read voltage array from an external file
-- Function to apply voltages to electrodes
function apply_voltages(voltages)
    for i, v in ipairs(voltages) do
        simion.wb.instances[i].adjustable_voltage = tonumber(v)
        print
    end
end

-- Main function that takes voltages from command line
function run_simulation()
    local args = {...}  -- Gets all command line arguments passed to the script
    table.remove(args, 1)  -- Remove the first argument, which is the script name
    apply_voltages(args)
    simion.run()  -- Runs the simulation
end

-- Call the main function
run_simulation()
