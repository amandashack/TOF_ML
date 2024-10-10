# config/generate_tf_config.py

import json
import os

def generate_tf_config(cluster_spec, task_type, task_index):
    tf_config = {
        'cluster': cluster_spec,
        'task': {'type': task_type, 'index': task_index}
    }
    return json.dumps(tf_config)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate TF_CONFIG for TensorFlow distributed training.')
    parser.add_argument('--task_type', type=str, required=True, help='Type of the task (e.g., worker, ps)')
    parser.add_argument('--task_index', type=int, required=True, help='Index of the task within its type')
    parser.add_argument('--config_file', type=str, default=None, help='Path to save the TF_CONFIG JSON string')

    args = parser.parse_args()

    # Define your cluster specification here
    cluster = {
        'worker': [
            'localhost:12345',
            'localhost:23456'
        ]
    }

    tf_config_json = generate_tf_config(cluster, args.task_type, args.task_index)

    if args.config_file:
        with open(args.config_file, 'w') as f:
            f.write(tf_config_json)
        print(f"TF_CONFIG written to {args.config_file}")
    else:
        # Set TF_CONFIG as an environment variable
        os.environ['TF_CONFIG'] = tf_config_json
        print("TF_CONFIG environment variable set.")
        print(tf_config_json)
