# src/data/utils.py

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from collections import defaultdict

def filter_data(
    data: np.ndarray,
    retardation_range=None,
    mid1=None,
    mid2=None,
    number_of_samples=None,
    random_state=42
) -> np.ndarray:
    """
    Applies optional filtering and sampling to the loaded data.
    Expects data of shape (N, 8) with the following columns:
      [0] = initial_ke
      [1] = initial_elevation
      [2] = mid1_ratio
      [3] = mid2_ratio
      [4] = retardation
      [5] = tof_values
      [6] = x_tof
      [7] = y_tof

    :param data: Numpy array of shape (N, 8)
    :param retardation_range: Optional [low, high] for filtering.
    :param mid1: Optional exact mid1 value to filter on (or None for no filter).
    :param mid2: Optional exact mid2 value to filter on (or None for no filter).
    :param number_of_samples: Optional integer limit for random subsampling.
    :param random_state: Random seed for shuffling during subsampling.
    :return: Filtered and/or sampled data.
    """

    # 1. Retardation range filter
    if retardation_range and len(retardation_range) == 2:
        low, high = retardation_range
        mask = (data[:, 4] >= low) & (data[:, 4] <= high)
        data = data[mask]

    # 2. mid1 filtering (if needed)
    if mid1 is not None and not isinstance(mid1, str):
        low, high = mid1
        data = data[(data[:, 2] >= low) & (data[:, 2] <= high)]

    # 3. mid2 filtering (if needed)
    if mid2 is not None and not isinstance(mid2, str):
        low, high = mid2
        data = data[(data[:, 3] >= low) & (data[:, 3] <= high)]

    # 4. Subsampling
    if number_of_samples and number_of_samples < len(data):
        data = shuffle(data, random_state=random_state)
        data = data[:number_of_samples]

    return data


