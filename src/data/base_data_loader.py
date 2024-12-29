# src/data/base_data_loader.py
from abc import ABC, abstractmethod
from typing import Any, Tuple
import pandas as pd


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    Each data loader should handle:
      - Reading raw data from a specified source (file, DB, etc.)
      - Optional initial preprocessing steps to produce a standardized data format
      - Returning data in a consistent interface (e.g., a DataFrame or a tuple of (X, y))
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
    def load_data(self) -> Any:
        """
        Load the data from the source and return it.
        The return type should be consistent across implementations:
          - Could return a pd.DataFrame or (X, y) tuple depending on your pipeline design.
        """
        pass

    @abstractmethod
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into train/test (and optionally validation) sets.
        Returns tuple: (X_train, X_test, y_train, y_test)
        """
        pass
