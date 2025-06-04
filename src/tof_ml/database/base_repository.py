#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base repository and query service interfaces for database abstraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple


class BaseRepository(ABC):
    """Abstract base class for database repositories (write operations)."""

    @abstractmethod
    def create_experiment(self, name: str, config: Dict[str, Any], description: str = "") -> str:
        """Create or get existing experiment."""
        pass

    @abstractmethod
    def create_run(self, experiment_id: str, config: Dict[str, Any], force_reload: bool = False) -> str:
        """Create a new run."""
        pass

    @abstractmethod
    def complete_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a run as completed."""
        pass

    @abstractmethod
    def store_artifact(self, run_id: str, file_path: str, artifact_type: str,
                       artifact_role: str, stage: str, metadata: Dict[str, Any] = None) -> str:
        """Store an artifact."""
        pass

    @abstractmethod
    def store_metrics(self, run_id: str, metrics: Dict[str, Any], stage: str,
                      epoch: int = None, metric_type: str = "predefined",
                      metric_category: str = "custom") -> None:
        """Store metrics."""
        pass

    @abstractmethod
    def store_parameters(self, run_id: str, parameters: Dict[str, Any], stage: str) -> None:
        """Store parameters for a run stage."""
        pass

    @abstractmethod
    def record_data_transformation(self, run_id: str, input_hash: str, output_hash: str,
                                   transformation_type: str, transformation_config: Dict[str, Any],
                                   stage: str) -> None:
        """Record a data transformation."""
        pass


class BaseQueryService(ABC):
    """Abstract base class for database query services (read operations)."""

    @abstractmethod
    def find_experiment_by_name_or_id(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Find experiment by name, ID, or partial match."""
        pass

    @abstractmethod
    def get_experiments(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of experiments."""
        pass

    @abstractmethod
    def get_runs_for_experiment(self, experiment_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get runs for a specific experiment."""
        pass

    @abstractmethod
    def get_run_lineage(self, run_id: str) -> Dict[str, Any]:
        """Get complete lineage for a run."""
        pass

    @abstractmethod
    def get_reusable_artifacts(self, experiment_id: str, pipeline_hash: str) -> Dict[str, str]:
        """Get reusable artifacts from previous runs."""
        pass

    @abstractmethod
    def get_training_progress(self, run_id: str) -> Dict[str, Any]:
        """Get real-time training progress for a run."""
        pass

    @abstractmethod
    def check_data_reuse(self, experiment_id: str, pipeline_hash: str, force_reload: bool) -> Tuple[bool, Optional[str]]:
        """Check if data can be reused from previous runs."""
        pass

    @abstractmethod
    def get_environment_info(self, env_hash: str) -> Optional[Dict[str, Any]]:
        """Get environment information by hash."""
        pass

    @abstractmethod
    def store_environment(self, env_hash: str, python_version: str, packages: str, system_info: str) -> None:
        """Store environment information."""
        pass

    @abstractmethod
    def get_recent_runs_for_experiment(self, experiment_id: str, current_run_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent runs for an experiment excluding current run."""
        pass

    @abstractmethod
    def get_run_status(self, run_id: str) -> Optional[str]:
        """Get the status of a run."""
        pass

    @abstractmethod
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        pass

    @abstractmethod
    def query_artifacts_by_criteria(self, artifact_type: str = None, stage: str = None,
                                    limit: int = 20) -> List[Dict[str, Any]]:
        """Query artifacts by criteria."""
        pass

    @abstractmethod
    def get_database_summary(self) -> Dict[str, Any]:
        """Get database summary statistics."""
        pass

    @abstractmethod
    def delete_run_cascade(self, run_id: str) -> bool:
        """Delete a run and all associated data."""
        pass
