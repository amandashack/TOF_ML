# scripts/launch_workers.py

import os
import subprocess
import json
import time

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

    # Launch the worker
    subprocess.Popen(cmd, env=env)

def main():
    # Define your cluster specification
    cluster_spec = {
        'worker': [
            'localhost:12345', 'localhost:23456'
        ]
    }

    training_script = os.path.abspath(os.path.join('train_model.py'))
    model_outpath = r"C:\Users\proxi\Documents\coding\stored_models\test_001\35"
    data_filepath = r"C:\Users\proxi\Documents\coding\TOF_data\TOF_data\combined_data.h5"
    params = {
        "layer_size": 32,
        "batch_size": int(1024/2),
        "dropout": 0.2,
        "learning_rate": 0.1,
        "optimizer": 'RMSprop',
        "job_name": "default_deep",  # Ensure this matches the handled cases
        "epochs": 10
    }
    param_ID = 12
    job_name = 'default_deep'
    sample_size = 200000

    num_workers = len(cluster_spec['worker'])

    processes = []
    for i in range(num_workers):
        launch_worker('worker', i, cluster_spec, training_script, model_outpath, data_filepath, params, param_ID, job_name, sample_size)
        time.sleep(1)  # Slight delay to prevent race conditions

    # Optionally, wait for all processes to finish
    # for p in processes:
    #     p.wait()

if __name__ == "__main__":
    main()
