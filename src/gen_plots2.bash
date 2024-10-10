#!/bin/bash

# Usage: ./generate_plots.sh /path/to/base_dir /path/to/data.h5

# Define the base directory where models are stored
BASE_DIR=$1
DATA_FILEPATH=$2  # Path to the data h5 file

# Path to the params file
PARAMS_FILE="$BASE_DIR/params"

# Loop over all directories in the base directory
for MODEL_DIR in "$BASE_DIR"/*; do
    MODEL_DIR_NAME=$(basename "$MODEL_DIR")

    # Check if the directory name matches the pattern
    if [[ "$MODEL_DIR_NAME" =~ ^([0-9]+)_(.*)$ ]]; then
        MODEL_ID=${BASH_REMATCH[1]}
        MODEL_TYPE=${BASH_REMATCH[2]}

        MODEL_PATH="$MODEL_DIR/main_model"

        if [ ! -d "$MODEL_PATH" ]; then
            echo "Model file not found in $MODEL_PATH"
        else
            # Read params for this model from the params file (if needed)
            # PARAMS_LINE=$(grep "^$MODEL_ID " "$PARAMS_FILE")
            # PARAMS_DICT=$(echo "$PARAMS_LINE" | cut -d' ' -f2-)

            # Generate the PDF plot for the model
            PDF_FILENAME="plots_model_${MODEL_ID}_${MODEL_TYPE}.pdf"
            python3 analyze_model.py "$BASE_DIR" "$MODEL_DIR_NAME" "$MODEL_TYPE" "$DATA_FILEPATH" --pdf_filename "$PDF_FILENAME"
        fi
    fi
done

echo "All plots generated."

