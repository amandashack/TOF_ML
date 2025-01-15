#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --job-name=tofs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20g
#SBATCH --time=0-24:00:00

# Activate the environment
# source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
# conda deactivate
# conda activate shack

DIR=$1
JOB_NAME=$2

PARAMS_FILE="${DIR}/params"
RESULTS_FILE="${DIR}/results"
RUNLOG_FILE="${DIR}/runlog"

if [ ! -d "$DIR" -o ! -f "$PARAMS_FILE" ]; then
    echo "Usage: $0 DIR JOB_NAME"
    echo "where DIR is a directory containing a file 'params' with the parameters."
    exit 1
fi

PARAMS_ID=$SLURM_ARRAY_TASK_ID
TOTAL_PARAMS=$(wc -l < "$PARAMS_FILE")

if [ "$PARAMS_ID" -gt "$TOTAL_PARAMS" ]; then
    echo "Error: PARAMS_ID ($PARAMS_ID) exceeds total number of parameters ($TOTAL_PARAMS)."
    exit 1
fi

JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "$PARAMS_ID|$JOB_ID|$SLURM_SUBMIT_DIR|$JOB_NAME" >> $RUNLOG_FILE

PARAMS=$(sed -n "${PARAMS_ID}p" "${PARAMS_FILE}")

echo "Setup tempfile"
TMPFILE=$(mktemp)
echo -n "$PARAMS_ID|$PARAMS|$JOB_NAME|" > $TMPFILE

echo "*** TRAIN ***"
MODEL_FILE="${DIR}/${PARAMS_ID}_${JOB_NAME}"
if [[ ! -d $MODEL_FILE ]]; then
    mkdir $MODEL_FILE
fi


python3 -m train_model "${MODEL_FILE}" "${JOB_NAME}" ${PARAMS} \
    | tee >(grep '^test_loss' | tr '\n\t' '| ' >> $TMPFILE)
echo >> $TMPFILE

# Exit if training failed
test $? -ne 0 && exit 1

# Append to the results file
cat $TMPFILE >> $RESULTS_FILE

# Cleanup
rm $TMPFILE

