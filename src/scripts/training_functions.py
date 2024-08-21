import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import h5py
import glob
#from loaders.load_and_save import DataGenerator, DataGeneratorWithVeto


def load_veto_model_if_exists(checkpoint_dir, fold):
    # Define the path to the directory above the checkpoint directory
    parent_dir = os.path.abspath(os.path.join(checkpoint_dir, os.pardir))
    veto_model_pattern = os.path.join(parent_dir, "veto_model_*.h5")

    # Search for any file that matches the pattern
    veto_model_files = glob.glob(veto_model_pattern)

    if veto_model_files:
        # If there are any matching files, load the first one (you can change this behavior if needed)
        veto_model_path = veto_model_files[0]
        print(f"Loading existing veto model from {veto_model_path}")
        return tf.keras.models.load_model(veto_model_path)
    else:
        print(f"No existing veto model found in {parent_dir}. A new one will be trained.")
        return None

