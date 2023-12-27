#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=milano
#SBATCH --job-name=tofs
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20g
#SBATCH --time=0-24:00:00

source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
conda deactivate
conda activate tf-gpu

# Ensure required dependencies are available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

BASE_DIR=$1
PARAMS_OFFSET=$2
USE_EXISTING_DIR=${3:-0}
N_HYPERPARAM_SETS=$4


# If USE_EXISTING_DIR is 0, create a new directory; otherwise, use the specified one
if [ "$USE_EXISTING_DIR" -eq 0 ]; then
    # Find the highest existing number in the directory names
    HIGHEST_NUMBER=0
    for DIR in "${BASE_DIR}/test_"*; do
        # Extract the numeric part of the directory name
        DIR_NUMBER="${DIR##*_}"
        # Check if it's a number and greater than the current highest number
        if [[ "$DIR_NUMBER" =~ ^[0-9]+$ && "$DIR_NUMBER" -gt "$HIGHEST_NUMBER" ]]; then
            HIGHEST_NUMBER="$DIR_NUMBER"
        fi
    done

    # Increment the highest number to get the next directory name
    NEXT_NUMBER=$((HIGHEST_NUMBER + 1))
    DIR="${BASE_DIR}/test_$(printf "%04d" "$NEXT_NUMBER")"
else
    DIR="${BASE_DIR}/test_$(printf "%04d" "$USE_EXISTING_DIR")"

    # Check if the params file exists in an existing directory
    PARAMS_FILE="${DIR}/params"
    if [ ! -f "$PARAMS_FILE" ]; then
        echo "Error: Params file not found in the existing directory."
        exit 1
    fi
fi

# Create the directory
mkdir -p "$DIR"

# Optional: Generate params file if N_HYPERPARAM_SETS is provided and it doesn't exist
if [ -n "$N_HYPERPARAM_SETS" ] && [ ! -f "$PARAMS_FILE" ]; then
    ./generate_params.py "$DIR" "$N_HYPERPARAM_SETS"
    PARAMS_FILE="${DIR}/params"
fi

# Check if the params file exists before proceeding
if [ ! -f "$PARAMS_FILE" ]; then
    echo "Error: Params file not found."
    exit 1
fi

RESULTS_FILE="${DIR}/results"
RUNLOG_FILE="${DIR}/runlog"


if [ -z "$PARAMS_OFFSET" ]
then
    PARAMS_OFFSET=0
fi

if [ ! -d "$DIR" -o ! -f "$PARAMS_FILE" ]
then
    echo "Usage: $0 DIR [PARAMS_OFFSET]"
    echo "where DIR is a directory containing a file 'params' with the parameters."
    exit 1
fi

PARAMS_ID=$(( $SLURM_ARRAY_TASK_ID + $PARAMS_OFFSET ))
JOB_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
BN="DEBUG"

echo "$PARAMS_ID|$JOB_NAME|$SLURM_SUBMIT_DIR" >> $RUNLOG_FILE

PARAMS=$(tail -n +${PARAMS_ID} ${PARAMS_FILE} | head -n 1)

echo "Setup tempfile"
# we assembled the needed data to a single line in $TMPFILE
TMPFILE=$(mktemp)
echo -n "$PARAMS_ID|$PARAMS|$JOB_NAME|$BN|" > $TMPFILE

echo "*** TRAIN ***"
MODEL_FILE="${DIR}/${PARAMS_ID}"
if [[ ! -d $MODEL_FILE ]] ; then
	mkdir $MODEL_FILE
fi
python3 simion_train.py ${MODEL_FILE} ${PARAMS} | tr '\n\t' '| ' >> $TMPFILE
echo >> $TMPFILE

# exit if training failed
test $? -ne 0 && exit 1

# only at the end we append it to the results file
cat $TMPFILE >> $RESULTS_FILE

# cleanup
rm $TMPFILE
