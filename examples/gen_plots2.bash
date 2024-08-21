#!/bin/bash

# Define the base directory where models are stored
BASE_DIR=$1

# Loop over all directories that are a number
for MODEL_DIR in "$BASE_DIR"/*; do
    MODEL_ID=$(basename "$MODEL_DIR")

    # Check if the directory name is a number
    if [[ "$MODEL_ID" =~ ^[0-9]+$ ]]; then
        MODEL_PATH="$BASE_DIR/$MODEL_ID"

        if [ ! -d "$MODEL_PATH" ]; then
            echo "There is something wrong with your model path: $MODEL_PATH"
        else
            # Combine the models from different folds
            python3 combine_folds.py "$BASE_DIR" "$MODEL_ID"

            # Generate the PDF plot for the combined model
            PDF_FILENAME="plots_model_${MODEL_ID}_combined.pdf"
            python3 analyze_surrogate.py "$BASE_DIR" "$MODEL_ID" --pdf_filename "$PDF_FILENAME"
        fi
    fi
done

echo "All plots generated."
