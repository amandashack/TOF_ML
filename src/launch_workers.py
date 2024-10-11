import os
import subprocess
import json
import time
import argparse

def launch_worker(task_type, task_index, cluster_spec, training_script, model_outpath,
                  data_filepath, params, param_ID, job_name, sample_size):
    tf_config = {
        'cluster': cluster_spec,
        'task': {'type': task_type, 'index': task_index}
    }

    env = os.environ.copy()
    env['TF_CONFIG'] = json.dumps(tf_config)
    # Disable GPUs
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    # Prepare the command
    cmd = [
        'python', training_script,
        '--model_outpath', model_outpath,
        '--data_filepath', data_filepath,
        '--param_ID', str(param_ID),
        '--job_name', job_name,
        '--sample_size', str(sample_size)
    ]

    # Convert params dictionary to JSON string
    params_json = json.dumps(params)
    env['PARAMS'] = params_json

    # Launch the worker and capture output
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def main(args):
    # Define your cluster specification
    cluster_spec = {
        'worker': [
            'localhost:12345', 'localhost:23456'
        ]
    }

    training_script = os.path.abspath('train_model.py')
    model_outpath = args.model_outpath
    data_filepath = args.data_filepath
    param_ID = args.param_ID
    job_name = args.job_name
    sample_size = args.sample_size
    params = {
        "layer_size": 32,
        "batch_size": int(1024/2),
        "dropout": 0.2,
        "learning_rate": 0.1,
        "optimizer": 'RMSprop',
        "job_name": job_name,
        "epochs": 10
    }

    num_workers = len(cluster_spec['worker'])

    processes = []
    for i in range(num_workers):
        p = launch_worker('worker', i, cluster_spec, training_script, model_outpath, data_filepath, params, param_ID, job_name, sample_size)
        processes.append(p)
        time.sleep(1)  # Slight delay to prevent race conditions

    # Wait for all processes to finish and capture their output
    for i, p in enumerate(processes):
        stdout, stderr = p.communicate()
        print(f"Worker {i} STDOUT:\n{stdout.decode()}")
        print(f"Worker {i} STDERR:\n{stderr.decode()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch TensorFlow workers.')
    parser.add_argument('--model_outpath', type=str, required=True)
    parser.add_argument('--data_filepath', type=str, required=True)
    parser.add_argument('--param_ID', type=int, required=True)
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=None)
    args = parser.parse_args()

    main(args)

