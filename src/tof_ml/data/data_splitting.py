from abc import ABC, abstractmethod
from typing import Optional, Dict
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class BaseDataSplitter(ABC):
    """
    Abstract base class for data splitting logic.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        column_mapping: Optional[dict] = None,
        random_state: Optional[int] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        meta_data: Optional[Dict] = None,
        **kwargs
    ):
        """
        If `config` is provided, we initialize from it first.
        Then override with direct parameters if not None.
        """
        # 1) Defaults
        self.config = {}
        self.column_mapping = {}
        self.random_state = 42
        self.test_size = 0.2
        self.val_size = 0.2
        self.meta_data = {} if meta_data is None else meta_data

        # 2) If we have a config, load from config
        if config:
            self._init_from_config(config, **kwargs)

        # 3) Override with direct parameters if not None
        if column_mapping is not None:
            self.column_mapping = column_mapping
        if random_state is not None:
            self.random_state = random_state
        if test_size is not None:
            self.test_size = test_size
        if val_size is not None:
            self.val_size = val_size

    def _init_from_config(self, config: Dict, **kwargs):
        """
        A helper for child classes. Reads fields from config or from
        "splitting" block if your config is nested. Adjust as needed.
        """
        self.config = config
        self.random_state = config.get("random_state", self.random_state)
        self.test_size = config.get("test_size", self.test_size)
        self.val_size = config.get("val_size", self.val_size)

    @abstractmethod
    def split(self, data: np.ndarray):
        """
        Must be overridden by subclasses.
        Should return (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        raise NotImplementedError

    def _update_metadata(self, data: np.ndarray, splits: dict):
        """
        Hook to store relevant info about how splitting was done.
        `splits` is a dict like:
          {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
          }
        This is where you can store shapes, summary stats, etc.
        """
        self.meta_data["splitter_class"] = self.__class__.__name__
        self.meta_data["random_state"] = self.random_state
        self.meta_data["test_size"] = self.test_size
        self.meta_data["val_size"] = self.val_size
        self.meta_data["original_data_shape"] = data.shape

        for split_name, arr in splits.items():
            if arr is not None:
                self.meta_data[f"{split_name}_shape"] = arr.shape

        # Optionally gather basic stats for each split (min, max, mean, etc.)
        # so you can later plot distributions in your report generator.
        # (This is optional, remove if not needed.)
        # self._store_basic_stats(splits)

    def _store_basic_stats(self, splits: dict):
        """
        Example: compute min, max, mean for each numeric column in each split
        and store in meta_data for later reporting.
        """
        # We'll store stats in a nested dict under "split_stats".
        # E.g., meta_data["split_stats"]["X_train"][<col_idx>] = dict(min=..., max=..., mean=...)
        self.meta_data.setdefault("split_stats", {})

        for split_name, arr in splits.items():
            if arr is None or arr.size == 0:
                continue
            # If X or y is 1D, reshape to 2D for consistent indexing
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            # Compute stats across columns
            min_vals = arr.min(axis=0)
            max_vals = arr.max(axis=0)
            mean_vals = arr.mean(axis=0)

            # Store in meta_data
            self.meta_data["split_stats"][split_name] = {
                col_idx: {
                    "min": float(min_vals[col_idx]),
                    "max": float(max_vals[col_idx]),
                    "mean": float(mean_vals[col_idx])
                }
                for col_idx in range(arr.shape[1])
            }


class UniformSplitter(BaseDataSplitter):
    """
    Simple uniform random split for train/val/test using sklearn's train_test_split.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        column_mapping: Optional[dict] = None,
        random_state: Optional[int] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        meta_data: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(
            config=config,
            column_mapping=column_mapping,
            random_state=random_state,
            test_size=test_size,
            val_size=val_size,
            meta_data=meta_data,
            **kwargs
        )

    def split(self, data: np.ndarray):
        """
        Expects 'data' to be shape (N, D+1) => the last column is target
        or rely on config["features"] with "input_columns" / "output_column".
        """
        # 1) Extract feature/target config
        input_cols = self.config.get("feature_columns", [])
        output_col = self.config.get("output_columns", 'initial_ke')

        # 2) Map feature names to indices
        feature_indices = [self.column_mapping[col] for col in input_cols]
        output_index = self.column_mapping[output_col]

        # 3) Slice X, y
        X = data[:, feature_indices]
        y = data[:, output_index]

        # 4) Split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state
        )

        logger.info(f"Training size: {X_train.shape}, "
                    f"Validation size: {X_val.shape}, "
                    f"Test size: {X_test.shape}")

        # 5) Update metadata
        splits_dict = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val":   X_val,
            "y_val":   y_val,
            "X_test":  X_test,
            "y_test":  y_test
        }
        self._update_metadata(data, splits_dict)

        return X_train, y_train, X_val, y_val, X_test, y_test


class SubsetSplitter(BaseDataSplitter):
    """
    Splitting approach that first extracts a subset by a specific column,
    trains on that subset, then merges remainder for val/test splits.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        column_mapping: Optional[dict] = None,
        random_state: Optional[int] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        meta_data: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(
            config=config,
            column_mapping=column_mapping,
            random_state=random_state,
            test_size=test_size,
            val_size=val_size,
            meta_data=meta_data,
            **kwargs
        )

    def split(self, data: np.ndarray):
        """
        1) Identify a 'subset_column', a set of 'subset_values'
        2) Extract that portion => train
        3) Merge remainder => val & test
        """
        features_config = self.config.get("features", {})
        input_cols = features_config.get("input_columns", [])
        output_col = features_config.get("output_column", None)

        subset_col_name = self.config.get("subset_column")
        subset_values = self.config.get("subset_values", [])
        train_fraction_for_subset = self.config.get("train_fraction_for_subset", 0.6)
        val_fraction = self.config.get("val_fraction", 0.2)
        test_fraction = self.config.get("test_fraction", 0.2)

        # 1) Indices
        feature_indices = [self.column_mapping[col] for col in input_cols]
        output_index = self.column_mapping[output_col]
        subset_idx = self.column_mapping[subset_col_name]

        # 2) Build subset
        subset_mask = np.isin(data[:, subset_idx], subset_values)
        subset_data = data[subset_mask]
        remainder_data = data[~subset_mask]

        logger.info(f"data size: {data.shape}, subset size: {subset_data.shape}, remainder: {remainder_data.shape}")

        X_subset = subset_data[:, feature_indices]
        y_subset = subset_data[:, output_index]

        # 3) train vs val+test from subset_data
        X_train_subset, X_valtest_subset, y_train_subset, y_valtest_subset = train_test_split(
            X_subset,
            y_subset,
            test_size=(1.0 - train_fraction_for_subset),
            random_state=self.random_state
        )

        # 4) Combine remainder_data with valtest_subset => a big pool for val/test
        combined_valtest = np.vstack([
            np.column_stack([X_valtest_subset, y_valtest_subset]),
            np.column_stack([remainder_data[:, feature_indices],
                             remainder_data[:, output_index]])
        ])
        X_combined = combined_valtest[:, :-1]
        y_combined = combined_valtest[:, -1]

        # 5) Val vs Test
        val_ratio_relative = val_fraction / (val_fraction + test_fraction)
        X_val_data, X_test_data, y_val_data, y_test_data = train_test_split(
            X_combined, y_combined,
            test_size=(1.0 - val_ratio_relative),
            random_state=self.random_state
        )

        # 6) Final splits
        X_train = X_train_subset
        y_train = y_train_subset
        X_val   = X_val_data
        y_val   = y_val_data
        X_test  = X_test_data
        y_test  = y_test_data

        logger.info(f"Training size: {X_train.shape}, "
                    f"Validation size: {X_val.shape}, "
                    f"Test size: {X_test.shape}")

        # 7) Update metadata
        splits_dict = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val":   X_val,
            "y_val":   y_val,
            "X_test":  X_test,
            "y_test":  y_test
        }

        # store subset info
        self.meta_data["subset_column"] = subset_col_name
        self.meta_data["subset_values"] = subset_values
        self.meta_data["train_fraction_for_subset"] = train_fraction_for_subset

        self._update_metadata(data, splits_dict)
        return X_train, y_train, X_val, y_val, X_test, y_test



def plot_split_distributions(
    data_before_split: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    column_mapping: dict,
    features_config: dict,
    plot_live: bool = False
):
    """
    Creates a 5 (rows) x 4 (columns) figure showing:
      Rows -> [retardation, mid1, mid2, initial_ke, tof]
      Cols -> [ENTIRE DATA, TRAIN, VAL, TEST]

    :param data_before_split: The entire data array (N, 8) BEFORE splitting.
    :param X_train, y_train, X_val, y_val, X_test, y_test: Post-split arrays.
    :param column_mapping: dict mapping feature/target names to their columns in 'data_before_split'.
    :param features_config: dict with 'input_columns' and 'output_column'.
    :param plot_live: If False, do nothing. If True, show the plot.
    """
    if not plot_live:
        return  # Skip plotting entirely

    # The columns we want for each row in the figure
    row_names = ['retardation', 'mid1', 'mid2', 'initial_ke', 'tof']
    nrows = len(row_names)
    ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3.5 * nrows))
    fig.suptitle("Distributions After Splitting", fontsize=16)

    # We'll handle the "bins" requirement for time-of-flight:
    # "make the number of bins the same as the number of unique points in initial KE".
    unique_ke = np.unique(data_before_split[:, column_mapping['initial_ke']])
    tof_bins = len(unique_ke)  # for row 'tof'

    # A helper to fetch column data from splitted sets:
    def get_column_data(col_name, X, y):
        """ Return an array of the requested column from the given splitted set (X or y). """
        # If col_name is the output_column, then the data is 'y'
        if col_name == features_config['output_column']:
            return y

        # Otherwise, col_name is in the input columns:
        input_idx = features_config['input_columns'].index(col_name)
        return X[:, input_idx]

    for row_idx, col_name in enumerate(row_names):
        # 1) Entire data
        data_col_index = column_mapping[col_name]  # e.g. column_mapping['retardation'] = 4
        entire_data_col = data_before_split[:, data_col_index]

        # 2) Train
        train_col = get_column_data(col_name, X_train, y_train)
        # 3) Val
        val_col   = get_column_data(col_name, X_val,   y_val)
        # 4) Test
        test_col  = get_column_data(col_name, X_test,  y_test)

        # For each column (0..3 in the figure):
        # col 0 -> entire data
        # col 1 -> train
        # col 2 -> val
        # col 3 -> test

        # Decide number of bins
        if col_name == 'tof':
            n_bins = tof_bins
        else:
            # A default choice, or you can choose something else
            n_bins = 50

        # Entire data
        ax_0 = axes[row_idx, 0] if nrows > 1 else axes[0]
        ax_0.hist(entire_data_col, bins=n_bins, color='gray', edgecolor='black')
        ax_0.set_title(f"All data ({col_name})")
        ax_0.set_ylabel("Count")

        # Train
        ax_1 = axes[row_idx, 1]
        ax_1.hist(train_col, bins=n_bins, color='blue', edgecolor='black')
        ax_1.set_title(f"Train ({col_name})")

        # Val
        ax_2 = axes[row_idx, 2]
        ax_2.hist(val_col, bins=n_bins, color='green', edgecolor='black')
        ax_2.set_title(f"Val ({col_name})")

        # Test
        ax_3 = axes[row_idx, 3]
        ax_3.hist(test_col, bins=n_bins, color='red', edgecolor='black')
        ax_3.set_title(f"Test ({col_name})")

        # Common x-label for each row
        axes[row_idx, 0].set_xlabel(col_name)
        axes[row_idx, 1].set_xlabel(col_name)
        axes[row_idx, 2].set_xlabel(col_name)
        axes[row_idx, 3].set_xlabel(col_name)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.show()
