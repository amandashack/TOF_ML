# src/data/preprocessor.py
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys

logger = logging.getLogger('trainer')

class DataPreprocessor:
    """
    Applies scaling, log transforms, and/or interaction terms to X, y.
    You can also incorporate old 'DataGenerator' logic if you like
    (calculating interactions, etc.).
    """

    def __init__(self, config: dict):
        """
        config might look like:
        {
          "apply_log_to_y": True,
          "scaler_type": "StandardScaler",
          "generate_interactions": True,
          ...
        }
        """
        self.config = config
        self.input_scaler = None
        self.fitted = False

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit the scalers on X (and/or y if you want) and transform them.
        Also apply log transform to y, create interactions, etc.
        """
        # 1) Possibly log-transform y
        logger.info("Take the log of both inputs and outputs.. ")
        logger.info("Here are the first 5 inputs and outputs BEFORE log: ",
                    X[:5, :].tolist(), y[:5, :].tolist())
        if self.config.get("apply_log", False) and (y is not None):
            # e.g. y = log2(y) if that’s desired
            y = np.log2(np.clip(y, 1e-9, None))
            X = np.log2(np.clip(X, 1e-9, None))

        logger.info("Here are the first 5 inputs and outputs AFTER log: ",
                    X[:5, :].tolist(), y[:5, :].tolist())
        # 2) Possibly create interactions for X
        logger.info("Generate interactions.. ")
        if self.config.get("generate_interactions", False):
            X = self._add_interactions(X)

        # 3) Scale X
        logger.info("Scaling.. ")
        scaler_type = self.config.get("scaler_type", "None")
        if scaler_type == "StandardScaler":
            self.input_scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            self.input_scaler = MinMaxScaler()
        else:
            self.input_scaler = None

        if self.input_scaler is not None:
            X = self.input_scaler.fit_transform(X)

        self.fitted = True
        return X, y

    def transform(self, X: np.ndarray, y: np.ndarray = None):
        """
        Transform new data after fitting.
        (If you do train/val/test splits *before* calling fit,
        then you want to call transform on val/test.)
        """
        if not self.fitted:
            raise ValueError("DataPreprocessor not fitted yet. Call fit_transform first.")

        # 1) Possibly log-transform y
        if self.config.get("apply_log_to_y", False) and (y is not None):
            y = np.log2(np.clip(y, 1e-9, None))

        # 2) Possibly create interactions
        if self.config.get("generate_interactions", False):
            X = self._add_interactions(X)

        # 3) Scale X
        if self.input_scaler is not None:
            X = self.input_scaler.transform(X)

        return X, y

    def inverse_transform(self, a: np.ndarray):
        """
        If you log-transformed y and X, do 2^y, 2^X to get back to original domain.
        """
        return 2 ** a

    def _add_interactions(self, X: np.ndarray):
        """
        Example interaction logic.
        You can import your older 'calculate_interactions' logic if you want deeper expansions.
        """
        X_copy = X.copy()
        # For demonstration, do pairwise (including squares):
        n_cols = X_copy.shape[1]
        features = [X_copy]  # start with original columns
        cross_terms = []
        for i in range(n_cols):
            for j in range(i, n_cols):
                product_ij = X_copy[:, i] * X_copy[:, j]
                cross_terms.append(product_ij.reshape(-1, 1))
        cross_array = np.hstack(cross_terms) if cross_terms else None
        if cross_array is not None:
            features.append(cross_array)

        return np.hstack(features)