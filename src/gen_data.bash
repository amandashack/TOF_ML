#!/bin/bash

# Define the base directory where simulation data is stored
BASE_DIR=$1
# define the voltage configuration you would like to run (for now this is just a retardation value)
voltage=$2
FLY_FILENAME=""
TOP_MODEL_IDS=$(python3 analyze_results.py ${BASE_DIR} --measure test_loss -N ${N_MODELS} --opt min --param_id)
echo $TOP_MODEL_IDS
# Loop over all model directories
for MODEL_ID in $TOP_MODEL_IDS; do
    MODEL_PATH="$BASE_DIR/$MODEL_ID"

    if [ ! -d "$MODEL_PATH" ]; then
        # Extract the model number from the directory name, assuming it follows "test_###"
        echo "There is something wrong with your model path"
    elif [ -d "$MODEL_PATH" ]; then
	# Run the Python script to generate the plots for this model
	python3 analyze_results.py "$BASE_DIR" --err_analysis "$MODEL_ID" --pdf_filename "$PDF_FILENAME"
    fi
done

echo "All plots generated."