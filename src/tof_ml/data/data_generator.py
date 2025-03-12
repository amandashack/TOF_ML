# src/tof_ml/data/data_generator.py
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """
    Example generator for in-memory data or partial reading from disk/HDF5.
    """
    def __init__(self, X, y, batch_size=32, shuffle=True):
        """
        X, y: entire dataset arrays or references
        batch_size: how many samples per batch
        shuffle: whether to shuffle indices on epoch end
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        """
        Number of batches per epoch.
        For example, if you have 1000 samples and batch_size=32,
        you get 31 full batches plus 1 partial batch => 32 total.
        """
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, index):
        """
        Generate one batch of data.
        index: batch index [0 .. __len__()-1]
        """
        # Figure out which slice of data belongs to this batch
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.X))

        # Grab the indices for this batch
        batch_indices = self.indexes[start_idx:end_idx]

        # Slice your data arrays
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        return X_batch, y_batch

    def on_epoch_end(self):
        """
        If you want to shuffle after each epoch, do it here.
        Keras calls this automatically after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)
