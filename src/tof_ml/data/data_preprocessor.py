# src/tof_ml/data/data_preprocessor.py

import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: Dict, column_mapping: Dict[str,int]):
        self.config = config
        self.column_mapping = column_mapping
        self.fitted_scaler = None
        self.scaler_type = None

    def mask_data(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies a mask if config['data']['mask_data'] is set.
        Returns (masked_data, step_info).
        """
        mask_col_name = self.config.get("data", {}).get("mask_data")
        if not mask_col_name or mask_col_name not in self.column_mapping:
            return data, {
                "operation": "mask_data",
                "skipped": True,
                "reason": f"No valid 'mask_data' found: {mask_col_name}"
            }

        original_size = data.shape[0]
        cidx = self.column_mapping[mask_col_name]
        yidx = self.column_mapping.get("y", None)
        print(cidx, yidx, np.min(data[:, cidx]), np.max(data[:, cidx]), np.min(data[:, yidx]), np.max(data[:, yidx]))
        if yidx is not None:
            print("yidx not None")
            mask = ((data[:, cidx].astype(bool)) & (data[:, yidx] > -16.2) & (data[:, yidx] < 16.2))
        else:
            mask = ((data[:, cidx] > 406) & (data[:, cidx] < 408))
        data = data[mask]

        step_info = {
            "operation": "mask_data",
            "mask_done": True,
            "original_size": original_size,
            "new_size": data.shape[0],
            "mask_col": mask_col_name
        }
        return data, step_info

    def apply_pass_energy(
        self, data: np.ndarray, pass_energy_config: bool, pass_energy_done: bool
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if pass_energy_done:
            return data, {
                "operation": "apply_pass_energy",
                "skipped": True,
                "reason": "pass_energy_done is already True"
            }
        if not pass_energy_config:
            return data, {
                "operation": "apply_pass_energy",
                "skipped": True,
                "reason": "Config says pass_energy=False"
            }
        ke_idx = self.column_mapping.get("kinetic_energy")
        ret_idx = self.column_mapping.get("retardation")
        if ke_idx is None or ret_idx is None:
            return data, {
                "operation": "apply_pass_energy",
                "skipped": True,
                "reason": "Missing kinetic_energy or retardation in column_mapping"
            }

        old_ke = data[:, ke_idx].copy()
        data[:, ke_idx] = data[:, ke_idx] - data[:, ret_idx]
        step_info = {
            "operation": "apply_pass_energy",
            "pass_energy_done": True,
            "sample_ke_before": old_ke[:5].tolist(),
            "sample_ke_after": data[:5, ke_idx].tolist()
        }
        return data, step_info

    def apply_log2(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str,Any]]:
        # Example: log transform X & y
        # We'll do naive log2(clip(x,1e-9,None))
        X_logged = np.log2(np.clip(X, 1e-9, None))
        y_logged = np.log2(np.clip(y, 1e-9, None))
        step_info = {
            "operation": "apply_log2",
            "log_done": True
        }
        return X_logged, y_logged, step_info

    def fit_transform_scaler(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaling_already_done: bool,
        force_rescale: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str,Any]]:
        """
        Fits a scaler if not already scaled or if force_rescale=True,
        returns (X_scaled, y_scaled, step_info).
        """
        if scaling_already_done and not force_rescale:
            return X, y, {
                "operation": "fit_transform_scaler",
                "skipped": True,
                "reason": "Already scaled, not forcing rescale"
            }

        # Example: read from config
        self.scaler_type = self.config.get("scaler", {}).get("type", None)
        if not self.scaler_type or self.scaler_type.lower() == "none":
            return X, y, {
                "operation": "fit_transform_scaler",
                "skipped": True,
                "reason": f"No scaler specified"
            }

        # Instantiate the scaler
        if self.scaler_type == "StandardScaler":
            from sklearn.preprocessing import StandardScaler
            self.fitted_scaler = StandardScaler()
        elif self.scaler_type == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler
            self.fitted_scaler = MinMaxScaler()
        else:
            return X, y, {
                "operation": "fit_transform_scaler",
                "skipped": True,
                "reason": f"Unknown scaler type: {self.scaler_type}"
            }

        old_sample = X[:5].copy()  # just to log some example
        X_scaled = self.fitted_scaler.fit_transform(X)
        step_info = {
            "operation": "fit_transform_scaler",
            "scaling_done": True,
            "scaler_params": self._extract_scaler_params(),
            "sample_X_before": old_sample.tolist(),
            "sample_X_after": X_scaled[:5].tolist()
        }
        # If you also want to scale y, do so here
        y_scaled = y  # or None
        return X_scaled, y_scaled, step_info

    def _extract_scaler_params(self) -> Dict[str, Any]:
        """Helper method to store fitted scaler params (scale_, mean_, etc.)."""
        params = {}
        if hasattr(self.fitted_scaler, "scale_"):
            params["scale_"] = self.fitted_scaler.scale_.tolist()
        if hasattr(self.fitted_scaler, "min_"):
            params["min_"] = self.fitted_scaler.min_.tolist()
        if hasattr(self.fitted_scaler, "data_min_"):
            params["data_min_"] = self.fitted_scaler.data_min_.tolist()
        if hasattr(self.fitted_scaler, "data_max_"):
            params["data_max_"] = self.fitted_scaler.data_max_.tolist()
        if hasattr(self.fitted_scaler, "mean_"):
            params["mean_"] = self.fitted_scaler.mean_.tolist()
        if hasattr(self.fitted_scaler, "var_"):
            params["var_"] = self.fitted_scaler.var_.tolist()
        return params
