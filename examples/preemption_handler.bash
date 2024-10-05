#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --job-name=tofs_preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=10g
#SBATCH --time=0-24:00:00

# Initialization
counter=1
max_retries=100
DIR=$1
PARAMS_OFFSET=$2
META_FILE="${DIR}/meta.txt"
job_script="run_bash2.bash"
PARAMS_FILE="${DIR}/params"
sleep_duration=1800  # Sleep duration in seconds

# Ensure the directory exists
if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist."
    exit 1
fi

# Initialize the meta file if it doesn't exist
if [[ ! -f $META_FILE ]]; then
    echo "Creating meta file at $META_FILE"
    touch $META_FILE
fi



# Submit the initial training jobs
sbatch_output=$(sbatch -a $PARAMS_OFFSET "$job_script" $DIR)
job_id=$(echo "$sbatch_output" | awk '{print $4}')
sleep $sleep_duration  # Wait before checking the job status

# Loop to monitor and handle preemption
while [ $counter -le $max_retries ]; do
    echo "Attempt: $counter"
    job_status=$(sacct -j $job_id --format=JobID,State --noheader | awk '{print $2}')

    if [[ "$job_status" == "PREEMPTED" ]]; then
        echo "Job was preempted. Resubmitting..."
        sbatch_output=$(sbatch -a $PARAMS_OFFSET "$job_script" $DIR)
        job_id=$(echo "$sbatch_output" | awk '{print $4}')
        sleep $sleep_duration
    elif [[ "$job_status" == "COMPLETED" ]]; then
        echo "Job completed successfully."
        break
    elif [[ "$job_status" == "FAILED" ]]; then
        echo "Job failed. Exiting."
        exit 1
    else
        echo "Job is still running."
        sleep $sleep_duration
    fi

    ((counter++))
done
