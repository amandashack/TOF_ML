# src/data/utils.py

import numpy as np
from sklearn.utils import shuffle

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
      [2] = x_tof
      [3] = y_tof
      [4] = mid1_ratio
      [5] = mid2_ratio
      [6] = retardation
      [7] = tof_values

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
        mask = (data[:, 6] >= low) & (data[:, 6] <= high)
        data = data[mask]

    # 2. mid1 filtering (if needed)
    if mid1 is not None and not isinstance(mid1, str):
        # If the user sets mid1=some_float, filter exactly or by tolerance
        # For exact match:
        data = data[np.isclose(data[:, 4], mid1)]

    # 3. mid2 filtering (if needed)
    if mid2 is not None and not isinstance(mid2, str):
        # Similarly, exact match or tolerance
        data = data[np.isclose(data[:, 5], mid2)]

    # 4. Subsampling
    if number_of_samples and number_of_samples < len(data):
        data = shuffle(data, random_state=random_state)
        data = data[:number_of_samples]

    return data
