#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --job-name=tofs
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus=3
#SBATCH --mem-per-cpu=10g
#SBATCH --time=0-24:00:00

# export TMPDIR=/scratch/<project>/tmp
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
conda deactivate
conda activate shack 
DIR=$1
PARAMS_OFFSET=$2


PARAMS_FILE="${DIR}/params"
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
python3 train_surrogate.py ${MODEL_FILE} ${PARAMS} | tr '\n\t' '| ' >> $TMPFILE
echo >> $TMPFILE

# exit if training failed
test $? -ne 0 && exit 1

# only at the end we append it to the results file
cat $TMPFILE >> $RESULTS_FILE

# cleanup
rm $TMPFILE
