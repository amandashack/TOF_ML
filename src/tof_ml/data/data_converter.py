import torch
import tensorflow as tf
import numpy as np

def to_torch_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

def to_tf_tensor(data):
    return tf.convert_to_tensor(data, dtype=tf.float32)

def to_sklearn(data):
    # If data is already a NumPy array, just return it
    return data
