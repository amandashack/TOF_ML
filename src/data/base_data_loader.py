from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np
import pandas as pd

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    Each data loader should:
      - read raw data from a specified source (file, DB, etc.)
      - parse or compute columns to produce an (N,8) array:
        [initial_ke, initial_elevation, x_tof, y_tof, mid1_ratio, mid2_ratio, retardation, tof_values]
      - split_data if needed (optional train/test/val splits).
    """

    def __init__(self, config: dict):
        """
        Initialize the data loader with a configuration dictionary.
        The config might contain:
          - file paths
          - column names
          - parameters for data splits
        """
        self.config = config

    @abstractmethod
    def load_data(self) -> np.ndarray:
        """
        Load the data from the source and return a shape (N, 8) numpy array.
        The columns must be:
        [initial_ke, initial_elevation, x_tof, y_tof, mid1_ratio, mid2_ratio, retardation, tof_values]
        """
        pass

    @abstractmethod
    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the data into train/test (and optionally validation) sets.
        Returns a tuple: (X_train, X_test, y_train, y_test), or another format as needed.
        """
        pass

