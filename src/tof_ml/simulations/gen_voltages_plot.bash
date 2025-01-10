#!/bin/bash

source "C:/ProgramData/Anaconda3/etc/profile.d/conda.sh"
conda activate arpes

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number of steps>"
    exit 1
fi

n=$1
output_file="simulation_results.txt"
echo "Retardation Front_Voltage Back_Voltage Max_Val OK" > "$output_file"

# Define the path where the output should be stored
output_path="C:/Users/proxi/Documents/coding/TOF_ML/simulations/ltspice"

retardations=(5 10 50 100)

for retardation in "${retardations[@]}"; do
    step=$(python -c "print(f'{float($retardation) * 0.10 / float($n):.2f}')")
    count=1
    for (( i=0; i<=$n; i++ )); do
        front_voltage=$(python -c "print(f'{float($i) * float($step):.2f}')")
        for (( j=0; j<=$n; j++ )); do
            back_voltage=$(python -c "print(f'{float($j) * float($step):.2f}')")
            # Format the filename with a zero-padded sequence number
            formatted_count=$(printf "%04d" $count)
            filename="${output_path}/voltage_${retardation}_${formatted_count}.raw"
            echo "Running with retardation=$retardation front_voltage=$front_voltage back_voltage=$back_voltage filename=$filename"
            output=$(python ltspice_runner.py "$retardation" "$filename" --front_voltage "$front_voltage" --back_voltage "$back_voltage")
            max_val=$(echo "$output" | cut -d ' ' -f1)
            ok=$(echo "$output" | cut -d ' ' -f2)
            echo "$retardation $front_voltage $back_voltage $max_val $ok" >> "$output_file"
            ((count++))
        done
    done
done

echo "Simulation complete. Results stored in $output_file."
