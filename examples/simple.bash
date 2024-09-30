#!/bin/bash

# This script runs train_surrogate.py with provided model output path and parameters
# Usage: ./simple_train.sh MODEL_OUTPUT_PATH param1=value1 param2=value2 ...

# Check if at least two arguments are provided (output path and at least one parameter)
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 MODEL_OUTPUT_PATH param1=value1 param2=value2 ..."
    exit 1
fi

MODEL_OUTPUT_PATH=$1
shift  # Remove the first argument to only leave the parameters

# Collect remaining arguments as parameters
PARAMS="$@"

# Call the Python script with model output path and parameters
python3 ../src/train_surrogate.py "${MODEL_OUTPUT_PATH}" ${PARAMS}