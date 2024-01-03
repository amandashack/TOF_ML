#!/bin/bash

# Define the base directory where models are stored
BASE_DIR="/Users/proxi/Documents/coding/TOF_ML/stored_models"

# Loop over all model directories
for model_dir in "$BASE_DIR"/test_*; do
  if [ -d "$model_dir" ]; then
    # Extract the model number from the directory name, assuming it follows "test_###"
    model_number=$(echo "$model_dir" | grep -o -E '[0-9]+')

    # Run the Python script to generate the plots for this model
    python analyze_results.py "$model_dir" --err_analysis "$model_number"
  fi
done

echo "All plots generated."