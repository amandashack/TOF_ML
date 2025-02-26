import numpy as np
from typing import Optional, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Applies scaling, log transforms, and/or interaction terms to X, y.
    Follows a pattern similar to loaders/splitters:
      - can read from config
      - direct parameters override config
      - maintains a meta_data dictionary for reporting (no live plotting here).
    """

    def __init__(
            self,
            config: Optional[Dict] = None,
            column_mapping: Optional[Dict] = None,
            # Potential direct overrides:
            log_transform: Optional[bool] = None,
            generate_interactions: Optional[bool] = None,
            scaler_type: Optional[str] = None,
            meta_data: Optional[Dict] = None,
            **kwargs
    ):
        """
        If `config` is provided, initialize from it first.
        Then override with direct parameters that are not None.
        """
        # 1) Set defaults
        self.config = {}
        self.log_transform = True
        self.generate_interactions = False
        self.scaler_type = None

        # We won't store or use plot_live since all plotting is removed.
        # self.plot_live = False  # no longer used

        self.column_mapping = column_mapping if column_mapping else {}
        self.meta_data = meta_data if meta_data is not None else {}
        self.input_scaler = None
        self.fitted = False

        # 2) Load from config if present
        if config is not None:
            self._init_from_config(config, **kwargs)

        # 3) Override with direct parameters
        if log_transform is not None:
            self.log_transform = log_transform
        if generate_interactions is not None:
            self.generate_interactions = generate_interactions
        if scaler_type is not None:
            self.scaler_type = scaler_type

    def _init_from_config(self, config: Dict, **kwargs):
        """
        Reads fields from config. Typically something like:
          {
            "data": {"log_transform": True},
            "features": {"generate_interactions": False},
            "scaler": {"type": "StandardScaler"}
          }
        Adjust to your actual schema.
        """
        self.config = config

        data_cfg = config.get("data", {})
        features_cfg = config.get("features", {})
        scaler_cfg = config.get("scaler", {})

        # Pull values
        self.log_transform = scaler_cfg.get("log_transform", self.log_transform)
        self.generate_interactions = features_cfg.get("interactions", False)
        self.scaler_type = scaler_cfg.get("type", self.scaler_type)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None):
        """
        1) Optionally apply log2 transform (X,y).
        2) Optionally create interaction terms in X.
        3) Fit a scaler (if specified) and transform X.

        Returns (X_trans, y_trans).
        """
        # 1) Log transform
        if self.log_transform and y is not None:
            X, y = self._apply_log_transform(X, y, dataset_name="Train")

        # 2) Generate interactions
        if self.generate_interactions:
            old_shape = X.shape
            X = self._add_interactions(X)
            logger.info(f"Generated interaction features: {old_shape} -> {X.shape}")

        # 3) Scale
        if self.scaler_type == "StandardScaler":
            self.input_scaler = StandardScaler()
        elif self.scaler_type == "MinMaxScaler":
            self.input_scaler = MinMaxScaler()
        else:
            self.input_scaler = None

        if self.input_scaler is not None:
            X = self.input_scaler.fit_transform(X)

        self.fitted = True

        # Update metadata after finishing training-phase transformations
        # self._update_metadata(X, y, dataset_name="Train")
        return X, y

    def transform(self, X: np.ndarray, y: np.ndarray = None, dataset_name="Val/Test"):
        """
        Applies the same transformations to new data (Val or Test):
          - log transform if self.log_transform
          - generate interactions if self.generate_interactions
          - apply the fitted scaler if self.input_scaler is not None
        """
        if not self.fitted:
            raise ValueError("DataPreprocessor not fitted yet. Call fit_transform first.")

        # 1) Log transform
        if self.log_transform and y is not None:
            X, y = self._apply_log_transform(X, y, dataset_name=dataset_name)

        # 2) Interactions
        if self.generate_interactions:
            old_shape = X.shape
            X = self._add_interactions(X)
            logger.info(f"Generated interaction features ({dataset_name}): {old_shape} -> {X.shape}")

        # 3) Scale
        if self.input_scaler is not None:
            X = self.input_scaler.transform(X)

        # Optionally update metadata for val/test
        # self._update_metadata(X, y, dataset_name=dataset_name)

        return X, y

    def _apply_log_transform(self, X: np.ndarray, y: np.ndarray, dataset_name="Train"):
        """
        Applies log2 transform to X and y (clip at 1e-9 to avoid log(0) or negative).
        """
        logger.info(f"Applying log2 transform for {dataset_name} set.")
        X_logged = np.log2(np.clip(X, 1e-9, None))
        y_logged = np.log2(np.clip(y, 1e-9, None))
        return X_logged, y_logged

    def _add_interactions(self, X: np.ndarray):
        """
        Example: generate pairwise products (including squares).
        """
        X_copy = X.copy()
        n_cols = X_copy.shape[1]
        cross_terms = []
        for i in range(n_cols):
            for j in range(i, n_cols):
                product_ij = X_copy[:, i] * X_copy[:, j]
                cross_terms.append(product_ij.reshape(-1, 1))

        if cross_terms:
            cross_array = np.hstack(cross_terms)
            return np.hstack([X_copy, cross_array])
        else:
            return X_copy

    def inverse_transform(self, arr: np.ndarray):
        """
        If you log-transformed data, do 2^arr to invert.
        If scaled, you might also call `self.input_scaler.inverse_transform(...)` for X.
        """
        return np.power(2, arr)

    def _update_metadata(self, X: np.ndarray, y: np.ndarray = None, dataset_name="Train"):
        """
        Record relevant transformation info in meta_data.
        Also optionally store basic stats so a report generator can later
        visualize the distribution changes.
        """
        # Basic transform flags
        self.meta_data[f"{dataset_name}_log_transform"] = self.log_transform
        self.meta_data[f"{dataset_name}_generate_interactions"] = self.generate_interactions
        self.meta_data[f"{dataset_name}_scaler_type"] = self.scaler_type

        # Optionally store shapes
        self.meta_data[f"X_{dataset_name}_shape"] = X.shape
        if y is not None and isinstance(y, np.ndarray):
            self.meta_data[f"y_{dataset_name}_shape"] = y.shape

        # (Optional) Store basic numeric stats for each column if you want
        self._store_data_stats(X, y, dataset_name)

    def _store_data_stats(self, X: np.ndarray, y: np.ndarray, dataset_name: str):
        """
        Example method to gather min, max, mean for X and y columns.
        You can expand or remove this as needed.
        """
        self.meta_data.setdefault("preprocessor_stats", {})
        stats_key = f"{dataset_name}_stats"
        self.meta_data["preprocessor_stats"].setdefault(stats_key, {})

        # X is 2D
        if X is not None and X.size > 0:
            min_vals = X.min(axis=0)
            max_vals = X.max(axis=0)
            mean_vals = X.mean(axis=0)
            col_stats = [
                {
                    "min": float(min_vals[i]),
                    "max": float(max_vals[i]),
                    "mean": float(mean_vals[i])
                }
                for i in range(X.shape[1])
            ]
            self.meta_data["preprocessor_stats"][stats_key]["X"] = col_stats

        # y is 1D
        if y is not None and y.size > 0:
            y_min, y_max, y_mean = float(y.min()), float(y.max()), float(y.mean())
            self.meta_data["preprocessor_stats"][stats_key]["y"] = {
                "min": y_min,
                "max": y_max,
                "mean": y_mean
            }

    def _columns_of_interest(self) -> List[str]:
        """
        Optionally, if you keep track of columns in the config, return them.
        (No plots happen here, but you may want them for external reference.)
        """
        data_config = self.config.get("data", {})
        input_cols = data_config.get("feature_columns", [])
        output_col = data_config.get("output_columns", None)
        return input_cols + ([output_col] if output_col else [])

    def _output_col_name(self) -> str:
        """
        If you ever need the output column name for referencing in meta_data or external reporting.
        """
        features_cfg = self.config.get("features", {})
        return features_cfg.get("output_column", "target")



