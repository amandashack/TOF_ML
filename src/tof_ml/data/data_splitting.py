# src/data/data_splitting.py
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from src.tof_ml.data.column_mapping import COLUMN_MAPPING

logger = logging.getLogger(__name__)

class BaseDataSplitter:
    """
    Abstract base class for data splitting logic using NumPy arrays.
    """
    def __init__(self, config: dict):
        self.config = config

    def split(self, data: np.ndarray):
        """
        Must be overridden by subclasses.
        Returns (X_train, y_train, X_val, y_val, X_test, y_test) in NumPy arrays.
        """
        raise NotImplementedError


class UniformSplitter(BaseDataSplitter):
    """
    Simple uniform random split for train/val/test using sklearn's train_test_split.
    """
    def __init__(self, config: dict, local_col_mapping: dict):
        super().__init__(config)
        self.local_col_mapping = local_col_mapping

    def split(self, data: np.ndarray):
        """
        Expects 'data' to be shape (N, D+1) => first D columns are features,
        last column is target. Or use config["features"] to figure out which columns
        are features vs. target if needed.
        """
        features_config = self.config["features"]
        input_cols = features_config["input_columns"]
        output_col = features_config["output_column"]

        # Map the feature names to indices
        feature_indices = [self.local_col_mapping[col] for col in input_cols]
        output_index = self.local_col_mapping[output_col]

        # If 'data' is already [X, y], you can skip. But let's assume data has all columns and we extract:
        X = data[:, feature_indices]
        y = data[:, output_index]

        # Now do train/val/test splits
        random_state = self.config.get("random_state", 42)
        test_size = self.config.get("test_size", 0.2)
        val_size = self.config.get("val_size", 0.2)  # or do something else

        # 1) train+val vs. test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 2) from X_temp, do val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )

        logger.info("Training size: ", X_train.shape, "\nValidation size: ", X_val.shape,
                    "\nTest size: ", X_test.shape)

        return X_train, y_train, X_val, y_val, X_test, y_test


class SubsetSplitter(BaseDataSplitter):
    def __init__(self, config: dict, local_col_mapping: dict):
        super().__init__(config)
        self.local_col_mapping = local_col_mapping

    def split(self, data: np.ndarray):
        # 1. Figure out which columns are features vs. target from config["features"]
        features_config = self.config["features"]
        input_cols = features_config["input_columns"]
        output_col = features_config["output_column"]

        feature_indices = [self.local_col_mapping[col] for col in input_cols]
        output_index = self.local_col_mapping[output_col]

        # 2. Identify the 'subset_column' (by name) -> get its index
        subset_col_name = self.config.get("subset_column")
        subset_idx = self.local_col_mapping[subset_col_name]

        subset_values = self.config.get("subset_values", [])
        train_fraction_for_subset = self.config.get("train_fraction_for_subset", 0.6)
        val_fraction = self.config.get("val_fraction", 0.2)
        test_fraction = self.config.get("test_fraction", 0.2)
        random_state = self.config.get("random_state", 42)

        # 3. Create mask for the subset
        #    e.g. all rows whose 'subset_idx' column in 'data' belongs to subset_values
        #    But careful: 'subset_idx' is among the global columns in data, not just the feature subset
        #    We must read from data[:, subset_idx]
        subset_mask = np.isin(data[:, subset_idx], subset_values)
        subset_data = data[subset_mask]
        remainder_data = data[~subset_mask]
        print("data size: ", data.shape, "\nsubset size: ", subset_data.shape,
                    "\nremainder size: ", remainder_data.shape)

        # 4. From subset_data, we only train on train_fraction_for_subset
        #    So do a train_test_split with test_size = (1 - fraction_for_subset)
        X_subset = subset_data[:, feature_indices]
        y_subset = subset_data[:, output_index]

        X_train_subset, X_valtest_subset, y_train_subset, y_valtest_subset = train_test_split(
            X_subset,
            y_subset,
            test_size=(1.0 - train_fraction_for_subset),
            random_state=random_state
        )

        # 5. Combine the remainder_data with the valtest_subset for the next split
        combined_valtest = np.vstack([ np.column_stack([X_valtest_subset, y_valtest_subset]),
                                       np.column_stack([
                                           remainder_data[:, feature_indices],
                                           remainder_data[:, output_index]
                                       ]) ])

        # At this point, combined_valtest is shape (M, D+1) => last col is target
        X_combined = combined_valtest[:, :-1]
        y_combined = combined_valtest[:, -1]

        # 6. We want val_fraction vs. test_fraction
        #    E.g. if val_fraction=0.2, test_fraction=0.2 => total is 0.4
        #    Then val_ratio_relative = val_fraction / (val_fraction + test_fraction)
        #    i.e. how big val is of the combined pool
        val_ratio_relative = val_fraction / (val_fraction + test_fraction)

        X_val_data, X_test_data, y_val_data, y_test_data = train_test_split(
            X_combined, y_combined,
            test_size=(1.0 - val_ratio_relative),
            random_state=random_state
        )

        # 7. Now we have:
        X_train = X_train_subset
        y_train = y_train_subset
        X_val   = X_val_data
        y_val   = y_val_data
        X_test  = X_test_data
        y_test  = y_test_data

        print("Training size: ", X_train.shape, "\nValidation size: ", X_val.shape,
                    "\nTest size: ", X_test.shape)

        return X_train, y_train, X_val, y_val, X_test, y_test


# For convenience in your pipeline, define the mapping:
SPLITTER_MAPPING = {
    "UniformSplitter": UniformSplitter,
    "SubsetSplitter": SubsetSplitter
}
