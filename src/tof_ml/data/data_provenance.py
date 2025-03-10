# src/tof_ml/data/data_provenance.py

import logging
import json
import h5py
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

@dataclass
class DataProvenance:
    raw_data: np.ndarray = field(default_factory=lambda: np.array([]))
    loader_metadata: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    column_mapping: Optional[Dict[str, Any]] = None
    # log that tracks each transformation step
    operations_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # 1) Merge all loader_metadata into self.metadata
        if self.loader_metadata:
            for k, v in self.loader_metadata.items():
                self.metadata[k] = v

        if self.config:
            for k, v in self.config.items():
                if k not in self.metadata:
                    self.metadata[k] = v

    def update_from_step(self, step_info: Dict[str, Any]):
        """
        Incorporate the step_info returned from a preprocessor step
        into this DataProvenance object.
        """
        self.operations_log.append(step_info)

        # If the step set pass_energy_done=True, you can reflect that in global metadata:
        if step_info.get("mask_done"):
            self.metadata["mask_done"] = step_info.get("mask_done")

        if step_info.get("pass_energy_done"):
            self.metadata["pass_energy_done"] = step_info.get("pass_energy_done")

        if step_info.get("scaling_done"):
            self.metadata["scaling_done"] = step_info.get("scaling_done")
            # If it has scaler params, store them
            params = step_info.get("params", {})
            self.metadata["scaler_params"] = params

        if step_info.get("log_done"):
            self.metadata["log_done"] = step_info.get("log_done")

        # If the step updated or replaced the column mapping
        if step_info.get("column_mapping"):
            self.column_mapping = step_info["column_mapping"]

    def write_to_h5(self, output_path: str, group_name="processed_data"):
        logger.info(f"Writing final data + provenance to {output_path}")
        with h5py.File(output_path, 'w') as f:
            grp = f.create_group(group_name)
            grp.create_dataset('data', data=self.raw_data)

            # store metadata
            for k,v in self.metadata.items():
                if isinstance(v, (dict, list)):
                    grp.attrs[k] = json.dumps(v)
                else:
                    grp.attrs[k] = v

    def write_operations_log_to_file(self, log_path: str):
        """
        Write the entire operations_log to a separate file (e.g., JSON).
        This way we don't clutter the HDF5 but still have a detailed record.
        """
        logger.info(f"Writing operations log to {log_path}")
        with open(log_path, 'w') as f:
            json.dump(self.operations_log, f, indent=2)
