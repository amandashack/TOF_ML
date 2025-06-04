#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Provenance Tracker with explicit data tracking and real-time metrics.
"""

import os
import logging
import functools
import time
import numpy as np
import pandas as pd
import contextlib
import hashlib
from typing import Dict, Any, Optional, Union, ContextManager, Callable

logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """
    Enhanced provenance tracker with explicit data tracking and database integration.
    """

    def __init__(self, config: Dict[str, Any], db_api, output_dir: str = None):
        """
        Initialize the enhanced provenance tracker.

        Args:
            config: Configuration dictionary
            db_api: Database API instance
            output_dir: Output directory for this run
        """
        self.config = config
        self.db_api = db_api
        self.output_dir = output_dir
        self.provenance_enabled = config.get("provenance", {}).get("enabled", True)

        # Initialize experiment and run
        self.experiment_name = config.get("experiment_name", "unnamed_experiment")
        self.experiment_id = None
        self.run_id = None

        # Current operation tracking
        self.current_stage = None
        self.stage_artifacts = {}
        self.stage_parameters = {}

        # Log file handling
        self.log_file_path = None

        # Model lineage tracking
        self.model_lineage_config = config.get("model", {}).get("lineage_tracking", {})
        self.model_lineage_enabled = self.model_lineage_config.get("enabled", True)
        self.model_lineage_frequency = self.model_lineage_config.get("frequency", "epoch")

        if self.provenance_enabled:
            self._initialize_experiment_and_run()
            # Set up log file AFTER we have run_id and output_dir
            self._setup_log_file()

        logger.info(f"ProvenanceTracker initialized for experiment '{self.experiment_name}'")

    def _initialize_experiment_and_run(self):
        """Initialize experiment and run in database."""
        # Create or get experiment (using stable ID generation)
        self.experiment_id = self.db_api.create_experiment(
            name=self.experiment_name,
            config=self.config,
            description=self.config.get("description", "")
        )

        logger.info(f"Experiment ID: {self.experiment_id}")

        # Create run (this should be unique for each run)
        force_reload = self.config.get("data", {}).get("force_reload", False)
        self.run_id = self.db_api.create_run(
            experiment_id=self.experiment_id,
            config=self.config,
            force_reload=force_reload
        )

        logger.info(f"Run ID: {self.run_id}")

    def _setup_log_file(self):
        """Set up log file for this run."""
        if not self.provenance_enabled or not self.run_id:
            return

        # Use the output directory from the orchestrator if available
        if self.output_dir:
            log_dir = os.path.join(self.output_dir, "logs")
        else:
            # Fallback to standard structure
            log_dir = f"./output/{self.experiment_name}/runs/{self.run_id}/logs"

        os.makedirs(log_dir, exist_ok=True)

        # Set up log file path
        self.log_file_path = os.path.join(log_dir, f"{self.run_id}.log")

        # Add file handler to root logger
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        logger.info(f"Log file initialized: {self.log_file_path}")

    def _record_log_artifact(self):
        """Record the log file as an artifact."""
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            return

        # Record log file as artifact
        log_hash = self.record_artifact(
            self.log_file_path,
            "log",
            "run_log",
            {
                "log_level": "INFO",
                "file_size": os.path.getsize(self.log_file_path)
            }
        )

        logger.info(f"Recorded log file artifact: {log_hash}")

    @contextlib.contextmanager
    def track_stage(self, stage_name: str, parameters: Dict[str, Any] = None) -> ContextManager:
        """Context manager to track a pipeline stage with metrics context."""
        if not self.provenance_enabled:
            yield
            return

        self.current_stage = stage_name
        self.stage_artifacts = {}
        self.stage_parameters = parameters or {}

        # Set metrics tracking context
        set_tracking_context(stage_name, self.run_id, self)

        # Record stage start
        start_time = time.time()
        logger.info(f"Started stage: {stage_name}")

        # Store stage parameters
        if self.stage_parameters:
            self.db_api.store_parameters(self.run_id, self.stage_parameters, stage_name)

        try:
            # Use metrics tracking context
            with MetricsTrackingContext(stage_name, self.run_id, self):
                yield self

            # Record successful completion
            end_time = time.time()
            duration = end_time - start_time

            logger.info(f"Completed stage: {stage_name} in {duration:.2f}s")

            # Store duration as metric
            self.record_metrics(
                {f"{stage_name}_duration_seconds": duration},
                metric_type="predefined",
                metric_category="performance"
            )

            # Record log file as artifact if this is the last stage
            if stage_name == "report_generation" and self.log_file_path:
                self._record_log_artifact()

        except Exception as e:
            logger.error(f"Failed stage: {stage_name} - {str(e)}")
            raise
        finally:
            clear_tracking_context()
            self.current_stage = None

    def record_artifact(self, file_path: str, artifact_type: str,
                        artifact_role: str, metadata: Dict[str, Any] = None) -> str:
        """Record an artifact with file existence validation."""
        if not self.provenance_enabled or not self.current_stage:
            logger.warning(f"Provenance not enabled or no current stage for artifact: {artifact_role}")
            return ""

        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Artifact file does not exist: {file_path}")
            return ""

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning(f"Artifact file is empty: {file_path}")

        logger.info(f"Recording artifact: {artifact_role} at {file_path} ({file_size} bytes)")

        # Calculate artifact hash
        artifact_hash = self._calculate_file_hash(file_path)

        try:
            # Store artifact using the enhanced database API
            self.db_api.store_artifact(
                run_id=self.run_id,
                file_path=file_path,
                artifact_type=artifact_type,
                artifact_role=artifact_role,
                stage=self.current_stage,
                metadata=metadata or {}
            )

            # Keep track for current stage
            self.stage_artifacts[artifact_role] = artifact_hash

            logger.info(f"Successfully recorded artifact {artifact_role}: {artifact_hash}")
            return artifact_hash

        except Exception as e:
            logger.error(f"Failed to store artifact {artifact_role}: {e}")
            return ""

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file or directory."""
        return self.db_api._calculate_file_hash(file_path)

    def record_data_transformation(self, input_artifact_hash: str,
                                   output_artifact_hash: str,
                                   transformation_type: str,
                                   transformation_config: Dict[str, Any]):
        """Record a data transformation."""
        if not self.provenance_enabled or not self.current_stage:
            return

        self.db_api.record_data_transformation(
            run_id=self.run_id,
            input_hash=input_artifact_hash,
            output_hash=output_artifact_hash,
            transformation_type=transformation_type,
            transformation_config=transformation_config,
            stage=self.current_stage
        )

    def record_metrics(self, metrics: Dict[str, Any], epoch: int = None,
                       metric_type: str = "predefined", metric_category: str = "custom"):
        """Record metrics for the current stage with enhanced categorization."""
        if not self.provenance_enabled or not self.current_stage:
            return

        self.db_api.store_metrics(
            run_id=self.run_id,
            metrics=metrics,
            stage=self.current_stage,
            epoch=epoch,
            metric_type=metric_type,
            metric_category=metric_category
        )

    def check_data_reuse(self) -> Dict[str, str]:
        """Check if data can be reused from previous runs."""
        if not self.provenance_enabled:
            return {}

        # Get pipeline hash for current config
        pipeline_hash = self.db_api._generate_pipeline_hash(self.config)

        logger.info(f"Checking data reuse for experiment: {self.experiment_id}")
        logger.info(f"Pipeline hash: {pipeline_hash}")

        # Check for reusable data artifacts
        reusable_artifacts = self.db_api.get_reusable_artifacts(
            self.experiment_id,
            pipeline_hash
        )

        logger.info(f"Reusable artifacts found: {reusable_artifacts}")
        return reusable_artifacts

    def get_artifact_path(self, artifact_hash: str) -> Optional[str]:
        """Get file path for an artifact hash."""
        if not self.provenance_enabled:
            return None

        cursor = self.db_api.connection.cursor()
        cursor.execute("SELECT file_path FROM artifacts WHERE hash_id = ?", (artifact_hash,))
        result = cursor.fetchone()

        return result["file_path"] if result else None

    def complete_run(self, status: str = "completed"):
        """Mark the current run as completed."""
        if not self.provenance_enabled:
            return

        # Record final log artifact if not already done
        if self.log_file_path and "run_log" not in self.stage_artifacts:
            self._record_log_artifact()

        self.db_api.complete_run(self.run_id, status)
        logger.info(f"Completed run {self.run_id} with status: {status}")

    def get_run_lineage(self) -> Dict[str, Any]:
        """Get complete lineage for current run."""
        if not self.provenance_enabled:
            return {}

        return self.db_api.get_run_lineage(self.run_id)

    def get_experiment_id(self) -> str:
        """Get current experiment ID."""
        return self.experiment_id or ""

    def get_run_id(self) -> str:
        """Get current run ID."""
        return self.run_id or ""

    def calculate_data_hash(self, data) -> str:
        """Calculate hash for data for provenance tracking."""
        if isinstance(data, pd.DataFrame):
            data_bytes = data.to_csv(index=False).encode()
        elif isinstance(data, tuple) and len(data) == 2:
            # Handle (X, y) tuple
            X, y = data
            try:
                if hasattr(X, 'tobytes'):
                    x_bytes = X.tobytes()
                elif hasattr(X, 'values') and hasattr(X.values, 'tobytes'):
                    x_bytes = X.values.tobytes()
                else:
                    x_bytes = str(X).encode()

                if hasattr(y, 'tobytes'):
                    y_bytes = y.tobytes()
                elif hasattr(y, 'values') and hasattr(y.values, 'tobytes'):
                    y_bytes = y.values.tobytes()
                else:
                    y_bytes = str(y).encode()

                data_bytes = x_bytes + y_bytes
            except:
                data_bytes = (str(X) + str(y)).encode()
        elif hasattr(data, 'tobytes'):
            data_bytes = data.tobytes()
        else:
            data_bytes = str(data).encode()

        return hashlib.sha256(data_bytes).hexdigest()

    # Legacy compatibility methods
    @property
    def run_dir(self) -> str:
        """Get run directory (legacy compatibility)."""
        return f"./output/{self.experiment_name}/runs/{self.run_id}"


# Global registry for tracking context
_tracking_context = {
    "current_stage": None,
    "current_run_id": None,
    "provenance_tracker": None
}


def set_tracking_context(stage: str, run_id: str, provenance_tracker):
    """Set the current tracking context."""
    _tracking_context.update({
        "current_stage": stage,
        "current_run_id": run_id,
        "provenance_tracker": provenance_tracker
    })


def clear_tracking_context():
    """Clear the current tracking context."""
    _tracking_context.update({
        "current_stage": None,
        "current_run_id": None,
        "provenance_tracker": None
    })


def get_current_stage() -> Optional[str]:
    """Get the current stage from context."""
    return _tracking_context.get("current_stage")


# Context manager for automatic stage tracking
class MetricsTrackingContext:
    """Context manager for automatic metrics tracking within a stage."""

    def __init__(self, stage: str, run_id: str, provenance_tracker):
        self.stage = stage
        self.run_id = run_id
        self.provenance_tracker = provenance_tracker
        self.previous_context = None

    def __enter__(self):
        # Store previous context
        self.previous_context = _tracking_context.copy()

        # Set new context
        set_tracking_context(self.stage, self.run_id, self.provenance_tracker)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        _tracking_context.update(self.previous_context)