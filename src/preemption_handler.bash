#!/bin/bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --job-name=tofs_preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10g
#SBATCH --time=0-24:00:00

# ---------------------------
# Initialization
# ---------------------------
# Assign command-line arguments to variables
DIR=$1
PARAMS_OFFSET=$2
JOB_NAME=$3

META_FILE="${DIR}/meta.txt"
JOB_SCRIPT="run_batch2.bash"
PARAMS_FILE="${DIR}/params"
SLEEP_DURATION=1800  # Sleep duration in seconds

# ---------------------------
# Usage Function
# ---------------------------
usage() {
    echo "Usage: $0 <DIR> <PARAMS_OFFSET> <JOB_NAME>"
    echo "Example: $0 test-1 3-7 tofs_job"
    exit 1
}

# ---------------------------
# Argument Validation
# ---------------------------
# Check if exactly three arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments."
    usage
fi

# Validate PARAMS_OFFSET Format and Extract Start and End
if ! [[ "$PARAMS_OFFSET" =~ ^[0-9]+-[0-9]+$ ]]; then
    echo "Error: PARAMS_OFFSET must be in the format start-end (e.g., 3-7)."
    exit 1
fi

START_OFFSET=$(echo "$PARAMS_OFFSET" | cut -d'-' -f1)
END_OFFSET=$(echo "$PARAMS_OFFSET" | cut -d'-' -f2)

# Ensure start is less than or equal to end
if [ "$START_OFFSET" -gt "$END_OFFSET" ]; then
    echo "Error: Start of PARAMS_OFFSET ($START_OFFSET) is greater than end ($END_OFFSET)."
    exit 1
fi

# Ensure Directory Exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

# Initialize the Meta File if it Doesn't Exist
if [[ ! -f "$META_FILE" ]]; then
    echo "Creating meta file at '$META_FILE'."
    touch "$META_FILE"
fi

# Update meta.txt with New param_IDs
new_entries=0

echo "Processing PARAMS_OFFSET range: $PARAMS_OFFSET"

for (( param_ID=START_OFFSET; param_ID<=END_OFFSET; param_ID++ ))
do
    if ! grep -q "^${param_ID}|${JOB_NAME}|" "$META_FILE"; then
        echo "${param_ID}|${JOB_NAME}|0" >> "$META_FILE"
        echo "Added to meta file: ${param_ID}|${JOB_NAME}|0"
        ((new_entries++))
    else
        echo "Entry already exists in meta file: ${param_ID}|${JOB_NAME}|0"
    fi
done

# Print Parameters for Verification
echo "Submission Details:"
echo "-------------------"
echo "Directory: $DIR"
echo "PARAMS_OFFSET: $PARAMS_OFFSET"
echo "JOB_NAME: $JOB_NAME"
echo "New Entries Added: $new_entries"
echo "-------------------"

# Submit the job array
sbatch_output=$(sbatch -a "$PARAMS_OFFSET" "$JOB_SCRIPT" "$DIR" "$JOB_NAME")
job_id=$(echo "$sbatch_output" | awk '{print $4}')

# Check if sbatch command was successful
if [ -z "$job_id" ]; then
    echo "Error: Failed to submit job."
    exit 1
fi

# Initialize an associative array to keep track of task IDs and their job IDs
declare -A task_job_ids

for (( param_ID=START_OFFSET; param_ID<=END_OFFSET; param_ID++ ))
do
    task_job_ids[$param_ID]=$job_id
done

counter=1
max_retries=100  # Set this to the maximum number of retries you want

while [ ${#task_job_ids[@]} -gt 0 ] && [ $counter -le $max_retries ]; do
    echo "Attempt: $counter"

    for task_id in "${!task_job_ids[@]}"; do
        current_job_id="${task_job_ids[$task_id]}"
        full_job_id="${current_job_id}_$task_id"

        # Check if the job is still running
        job_running=$(squeue -j $full_job_id -h -o %T)

        if [ -n "$job_running" ]; then
            echo "Task $full_job_id is still running."
            # No action needed
        else
            # Job is not in squeue; get the job status from sacct
            job_status=$(sacct -n -P -j $full_job_id --format=JobID,State | grep "^$full_job_id|" | tail -1 | awk -F'|' '{print $2}')

            echo "Task $full_job_id Status: $job_status"

            if echo "$job_status" | grep -q "COMPLETED"; then
                echo "Task $full_job_id completed successfully."
                unset task_job_ids[$task_id]
            elif echo "$job_status" | grep -q "PREEMPTED"; then
                echo "Task $full_job_id was preempted. Resubmitting task $task_id..."
                sbatch_output=$(sbatch -a "$task_id" "$JOB_SCRIPT" "$DIR" "$JOB_NAME")
                new_job_id=$(echo "$sbatch_output" | awk '{print $4}')
                if [ -z "$new_job_id" ]; then
                    echo "Error: Failed to resubmit task $task_id."
                    exit 1
                fi
                task_job_ids[$task_id]=$new_job_id
            elif echo "$job_status" | grep -q "FAILED"; then
                echo "Task $full_job_id failed. Exiting."
                exit 1
            else
                echo "Task $full_job_id has unknown status: $job_status"
                # Decide what to do
            fi
        fi
    done

    if [ ${#task_job_ids[@]} -eq 0 ]; then
        echo "All tasks completed."
        break
    fi

    sleep $SLEEP_DURATION
    ((counter++))
done

if [ $counter -gt $max_retries ]; then
    echo "Maximum number of retries ($max_retries) exceeded."
    exit 1
fi

