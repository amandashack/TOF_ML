#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Provenance Tracker for the ML Provenance Tracker Framework.
This module tracks data lineage and transformations.
"""

import os
import logging
import json
import time
import datetime
import uuid
import contextlib
import copy
from typing import Dict, Any, Optional, List, Tuple, Union, ContextManager

logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """
    Tracks data provenance throughout the ML pipeline.

    This class records:
    - Data sources and their metadata
    - Transformations applied to data
    - Data splits
    - Model training inputs
    - Full lineage (data -> transformations -> models -> outputs)
    - Stage completion and artifacts
    - Experiment and run metadata
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the provenance tracker.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.provenance_enabled = config.get("provenance", {}).get("enabled", True)

        # Get experiment name from config
        self.experiment_name = config.get("experiment_name", "unnamed_experiment")

        # Base output directory
        base_output_dir = config.get("output_dir", "./output")

        # Experiment directory (shared across runs)
        self.experiment_dir = os.path.join(base_output_dir, self.experiment_name)

        # Generate run ID with timestamp for better readability
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        self.start_time = datetime.datetime.now().isoformat()

        # Run-specific directory
        self.run_dir = os.path.join(self.experiment_dir, "runs", self.run_id)

        # Set provenance directory for this experiment
        self.provenance_db_path = os.path.join(self.experiment_dir, "provenance")

        # Create provenance storage
        self.data_sources = {}
        self.transformations = {}
        self.data_splits = {}
        self.operations = {}
        self.lineage_graph = {}

        # Create run record with pipeline configuration hash
        self.run_record = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "config": self.config.copy(),
            "pipeline_hash": self._generate_pipeline_hash(self.config),
            "stages": {}
        }

        # Create necessary directories
        if self.provenance_enabled:
            os.makedirs(self.experiment_dir, exist_ok=True)
            os.makedirs(os.path.join(self.experiment_dir, "runs"), exist_ok=True)
            os.makedirs(self.run_dir, exist_ok=True)
            os.makedirs(os.path.join(self.run_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(self.run_dir, "data"), exist_ok=True)
            os.makedirs(os.path.join(self.run_dir, "reports"), exist_ok=True)
            os.makedirs(self.provenance_db_path, exist_ok=True)
            os.makedirs(os.path.join(self.provenance_db_path, "runs"), exist_ok=True)
            os.makedirs(os.path.join(self.provenance_db_path, "data_sources"), exist_ok=True)
            os.makedirs(os.path.join(self.provenance_db_path, "transformations"), exist_ok=True)
            os.makedirs(os.path.join(self.provenance_db_path, "data_splits"), exist_ok=True)
            os.makedirs(os.path.join(self.provenance_db_path, "operations"), exist_ok=True)

            # Save initial run record
            self._save_run_record()

        logger.info(f"ProvenanceTracker initialized for experiment '{self.experiment_name}', run '{self.run_id}'")

    def _generate_pipeline_hash(self, config: Dict[str, Any]) -> str:
        """
        Generate a hash of the pipeline configuration.

        Args:
            config: Pipeline configuration dictionary

        Returns:
            Hash string representing the pipeline configuration
        """
        # Extract parts of the config that affect data processing
        pipeline_config = self._extract_pipeline_config(config)

        # Create a deterministic string representation
        import hashlib

        # Sort keys to ensure deterministic serialization
        config_str = json.dumps(pipeline_config, sort_keys=True)

        # Generate hash
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _extract_pipeline_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize the pipeline configuration parts that affect data processing."""
        # Deep copy relevant config sections
        pipeline_config = {
            "data": copy.deepcopy(config.get("data", {})),
            "preprocessing": copy.deepcopy(config.get("preprocessing", {})),
            "data_splitting": copy.deepcopy(config.get("data_splitting", {}))
        }

        # Remove fields that don't affect processing
        if "force_reload" in pipeline_config["data"]:
            del pipeline_config["data"]["force_reload"]

        if "source_run_id" in pipeline_config["data"]:
            del pipeline_config["data"]["source_run_id"]

        return pipeline_config

    def _save_run_record(self):
        """Save the current run record to the provenance database."""
        if not self.provenance_enabled:
            return

        # Update the run record with current time
        self.run_record["last_updated"] = datetime.datetime.now().isoformat()

        # Save to provenance/runs directory
        run_record_path = os.path.join(self.provenance_db_path, "runs", f"{self.run_id}.json")
        with open(run_record_path, 'w') as f:
            json.dump(self.run_record, f, indent=2, default=str)

    def update_run_record(self, updates: Dict[str, Any]):
        """
        Update the run record with new information.

        Args:
            updates: Dictionary of updates to apply to the run record
        """
        if not self.provenance_enabled:
            return

        # Update the run record
        self.run_record.update(updates)

        # Save the updated record
        self._save_run_record()

        logger.debug(f"Updated run record for {self.run_id}")

    @contextlib.contextmanager
    def track_operation(self, operation_name: str) -> ContextManager:
        """
        Context manager to track an operation's execution time and metadata.

        Args:
            operation_name: Name of the operation to track

        Yields:
            Context for the operation
        """
        if not self.provenance_enabled:
            yield
            return

        # Generate operation ID
        operation_id = f"{operation_name}_{uuid.uuid4()}"

        # Record start
        start_time = time.time()
        logger.debug(f"Starting operation: {operation_name} (id: {operation_id})")

        try:
            # Yield control back to the caller
            yield

            # Record success
            status = "completed"
            error = None
        except Exception as e:
            # Record failure
            status = "failed"
            error = str(e)
            raise
        finally:
            # Record end time and duration
            end_time = time.time()
            duration = end_time - start_time

            # Store operation metadata
            self.operations[operation_id] = {
                "operation_name": operation_name,
                "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.datetime.fromtimestamp(end_time).isoformat(),
                "duration_seconds": duration,
                "status": status,
                "error": error,
                "run_id": self.run_id
            }

            # Record in provenance database
            self._save_provenance_record("operations", operation_id, self.operations[operation_id])

            logger.debug(f"Finished operation: {operation_name} in {duration:.2f}s (status: {status})")

    def update_stage_completion(self, stage_name: str, success: bool = True, artifacts: Dict[str, str] = None):
        """
        Update the run record with stage completion information.

        Args:
            stage_name: Name of the completed stage (e.g., 'data_loading', 'preprocessing', 'splitting', 'training')
            success: Whether the stage was successful
            artifacts: Dictionary of artifacts (files) produced by this stage
        """
        if not self.provenance_enabled:
            return

        # Update stage information
        self.run_record["stages"][stage_name] = {
            "completed": success,
            "timestamp": datetime.datetime.now().isoformat(),
            "artifacts": artifacts or {}
        }

        # Save the updated record
        self._save_run_record()

        logger.debug(f"Updated stage completion for {stage_name} in run {self.run_id}")

    def record_data_source(self, data_id: str, source_info: Dict[str, Any]) -> None:
        """
        Record information about a data source.

        Args:
            data_id: Unique identifier for the data
            source_info: Information about the data source
        """
        if not self.provenance_enabled:
            return

        logger.debug(f"Recording data source: {data_id}")

        # Store data source information
        self.data_sources[data_id] = {
            "data_id": data_id,
            "source_info": source_info,
            "run_id": self.run_id
        }

        # Add to lineage graph
        self.lineage_graph[data_id] = {
            "type": "data_source",
            "outputs": []
        }

        # Save to provenance database
        self._save_provenance_record("data_sources", data_id, self.data_sources[data_id])

    def record_transformation(
            self,
            input_data_id: str,
            output_data_id: str,
            transformation_info: Dict[str, Any]
    ) -> None:
        """
        Record information about a data transformation.

        Args:
            input_data_id: ID of the input data
            output_data_id: ID of the output data
            transformation_info: Information about the transformation
        """
        if not self.provenance_enabled:
            return

        logger.debug(f"Recording transformation: {input_data_id} -> {output_data_id}")

        # Generate transformation ID
        transformation_id = f"transform_{uuid.uuid4()}"

        # Store transformation information
        self.transformations[transformation_id] = {
            "transformation_id": transformation_id,
            "input_data_id": input_data_id,
            "output_data_id": output_data_id,
            "transformation_info": transformation_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": self.run_id
        }

        # Update lineage graph
        if input_data_id in self.lineage_graph:
            self.lineage_graph[input_data_id]["outputs"].append({
                "id": output_data_id,
                "via_transformation": transformation_id
            })

        self.lineage_graph[output_data_id] = {
            "type": "transformed_data",
            "inputs": [{
                "id": input_data_id,
                "via_transformation": transformation_id
            }],
            "outputs": []
        }

        # Save to provenance database
        self._save_provenance_record("transformations", transformation_id, self.transformations[transformation_id])

    def record_data_split(
            self,
            input_data_id: str,
            train_data_id: str,
            val_data_id: str,
            test_data_id: str,
            split_info: Dict[str, Any]
    ) -> None:
        """
        Record information about a data split operation.

        Args:
            input_data_id: ID of the input data
            train_data_id: ID of the training data
            val_data_id: ID of the validation data
            test_data_id: ID of the test data
            split_info: Information about the split operation
        """
        if not self.provenance_enabled:
            return

        logger.debug(f"Recording data split: {input_data_id} -> {train_data_id}, {val_data_id}, {test_data_id}")

        # Generate split ID
        split_id = f"split_{uuid.uuid4()}"

        # Store split information
        self.data_splits[split_id] = {
            "split_id": split_id,
            "input_data_id": input_data_id,
            "split_info": split_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": self.run_id
        }

        # Update lineage graph
        if input_data_id in self.lineage_graph:
            self.lineage_graph[input_data_id]["outputs"].extend([
                {"id": train_data_id, "via_split": split_id, "split_type": "train"},
                {"id": val_data_id, "via_split": split_id, "split_type": "validation"},
                {"id": test_data_id, "via_split": split_id, "split_type": "test"}
            ])

        # Add split outputs to lineage graph
        for data_id, split_type in [
            (train_data_id, "train"),
            (val_data_id, "validation"),
            (test_data_id, "test")
        ]:
            self.lineage_graph[data_id] = {
                "type": f"{split_type}_data",
                "inputs": [{
                    "id": input_data_id,
                    "via_split": split_id,
                    "split_type": split_type
                }],
                "outputs": []
            }

        # Save to provenance database
        self._save_provenance_record("data_splits", split_id, self.data_splits[split_id])

        # Update stage completion with artifacts
        artifacts = {
            "train_data_id": train_data_id,
            "val_data_id": val_data_id,
            "test_data_id": test_data_id
        }
        self.update_stage_completion("data_splitting", True, artifacts)

    def record_model_training(
            self,
            model_id: str,
            train_data_id: str,
            val_data_id: str,
            model_info: Dict[str, Any],
            training_info: Dict[str, Any]
    ) -> None:
        """
        Record information about model training.

        Args:
            model_id: ID of the trained model
            train_data_id: ID of the training data
            val_data_id: ID of the validation data
            model_info: Information about the model
            training_info: Information about the training process
        """
        if not self.provenance_enabled:
            return

        logger.debug(f"Recording model training: {train_data_id}, {val_data_id} -> {model_id}")

        # Generate training ID
        training_id = f"training_{uuid.uuid4()}"

        # Store training information
        self.operations[training_id] = {
            "operation_id": training_id,
            "operation_name": "model_training",
            "model_id": model_id,
            "train_data_id": train_data_id,
            "val_data_id": val_data_id,
            "model_info": model_info,
            "training_info": training_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": self.run_id
        }

        # Update lineage graph
        if train_data_id in self.lineage_graph:
            self.lineage_graph[train_data_id]["outputs"].append({
                "id": model_id,
                "via_training": training_id,
                "role": "training_data"
            })

        if val_data_id in self.lineage_graph:
            self.lineage_graph[val_data_id]["outputs"].append({
                "id": model_id,
                "via_training": training_id,
                "role": "validation_data"
            })

        self.lineage_graph[model_id] = {
            "type": "trained_model",
            "inputs": [
                {"id": train_data_id, "via_training": training_id, "role": "training_data"},
                {"id": val_data_id, "via_training": training_id, "role": "validation_data"}
            ],
            "outputs": []
        }

        # Save to provenance database
        self._save_provenance_record("operations", training_id, self.operations[training_id])

        # Update stage completion with artifacts
        artifacts = {
            "model_id": model_id,
            "model_path": training_info.get("best_model_path", ""),
            "final_model_path": training_info.get("final_model_path", "")
        }
        self.update_stage_completion("model_training", True, artifacts)

    def record_model_evaluation(
            self,
            model_id: str,
            test_data_id: str,
            evaluation_results: Dict[str, Any]
    ) -> None:
        """
        Record information about model evaluation.

        Args:
            model_id: ID of the model
            test_data_id: ID of the test data
            evaluation_results: Results of the evaluation
        """
        if not self.provenance_enabled:
            return

        logger.debug(f"Recording model evaluation: {model_id} on {test_data_id}")

        # Generate evaluation ID
        evaluation_id = f"evaluation_{uuid.uuid4()}"

        # Store evaluation information
        self.operations[evaluation_id] = {
            "operation_id": evaluation_id,
            "operation_name": "model_evaluation",
            "model_id": model_id,
            "test_data_id": test_data_id,
            "evaluation_results": evaluation_results,
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": self.run_id
        }

        # Add evaluation results to model in lineage graph
        if model_id in self.lineage_graph:
            self.lineage_graph[model_id]["evaluation"] = {
                "evaluation_id": evaluation_id,
                "test_data_id": test_data_id,
                "results": evaluation_results
            }

        if test_data_id in self.lineage_graph:
            self.lineage_graph[test_data_id]["outputs"].append({
                "id": model_id,
                "via_evaluation": evaluation_id,
                "role": "test_data"
            })

        # Save to provenance database
        self._save_provenance_record("operations", evaluation_id, self.operations[evaluation_id])

        # Update stage completion with artifacts
        artifacts = {
            "evaluation_results_path": os.path.join(self.run_dir, "evaluation_results.json")
        }
        self.update_stage_completion("model_evaluation", True, artifacts)

    def _save_provenance_record(self, record_type: str, record_id: str, record_data: Dict[str, Any]) -> None:
        """Save a provenance record to disk."""
        if not self.provenance_enabled:
            return

        # Create record type directory if it doesn't exist
        record_type_dir = os.path.join(self.provenance_db_path, record_type)
        os.makedirs(record_type_dir, exist_ok=True)

        # Save record to disk
        record_path = os.path.join(record_type_dir, f"{record_id}.json")
        with open(record_path, 'w') as f:
            json.dump(record_data, f, indent=2, default=str)

    def get_experiment_runs(self) -> List[Dict[str, Any]]:
        """
        Get information about all runs for the current experiment.

        Returns:
            List of run records sorted by start time (newest first)
        """
        runs = []
        runs_dir = os.path.join(self.provenance_db_path, "runs")

        if not os.path.exists(runs_dir):
            return runs

        for filename in os.listdir(runs_dir):
            if filename.endswith(".json"):
                with open(os.path.join(runs_dir, filename), 'r') as f:
                    run_record = json.load(f)
                    runs.append(run_record)

        # Sort by start_time (newest first)
        runs.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return runs

    def get_latest_run(self, exclude_current: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the latest run record for this experiment.

        Args:
            exclude_current: Whether to exclude the current run

        Returns:
            The latest run record or None if no previous runs
        """
        runs = self.get_experiment_runs()

        # Filter out current run if requested
        if exclude_current:
            runs = [r for r in runs if r.get("run_id") != self.run_id]

        return runs[0] if runs else None

    def get_run_dir(self, run_id: Optional[str] = None) -> str:
        """
        Get the directory path for a specific run.

        Args:
            run_id: ID of the run (uses current run if None)

        Returns:
            Path to the run directory
        """
        if run_id is None or run_id == self.run_id:
            return self.run_dir

        return os.path.join(self.experiment_dir, "runs", run_id)

    def get_data_path(self, run_id: Optional[str] = None, dataset_type: str = "train") -> Optional[str]:
        """
        Get the path to a specific dataset for a run.

        Args:
            run_id: ID of the run (uses current run if None)
            dataset_type: Type of dataset ('train', 'val', or 'test')

        Returns:
            Path to the dataset or None if not found
        """
        if run_id is None:
            run_id = self.run_id

        run_dir = self.get_run_dir(run_id)
        dataset_path = os.path.join(run_dir, "data", f"{dataset_type}_data.h5")

        return dataset_path if os.path.exists(dataset_path) else None

    def find_best_run_for_stage(self, stage_name: str, required_artifact: str = None) -> Optional[Dict[str, Any]]:
        """
        Find the best run that has successfully completed a specific stage.

        Args:
            stage_name: Name of the stage (e.g., 'data_splitting')
            required_artifact: Optional specific artifact that must be present

        Returns:
            Run record of the best run, or None if no suitable run found
        """
        runs = self.get_experiment_runs()

        # Filter runs that have successfully completed the stage
        valid_runs = []
        for run in runs:
            # Check if the stage was completed successfully
            stages = run.get("stages", {})
            if stage_name in stages and stages[stage_name].get("completed", False):
                # Check if required artifact exists
                if required_artifact:
                    artifacts = stages[stage_name].get("artifacts", {})
                    if required_artifact in artifacts and os.path.exists(artifacts[required_artifact]):
                        valid_runs.append(run)
                else:
                    valid_runs.append(run)

        # Sort by timestamp (newest first) and return the best match
        valid_runs.sort(key=lambda x: x.get("stages", {}).get(stage_name, {}).get("timestamp", ""), reverse=True)
        return valid_runs[0] if valid_runs else None

    def get_stage_artifacts(self, run_id: str, stage_name: str) -> Dict[str, str]:
        """
        Get the artifacts produced by a specific stage in a specific run.

        Args:
            run_id: ID of the run
            stage_name: Name of the stage

        Returns:
            Dictionary of artifact names to paths
        """
        # Get the run record
        run_record_path = os.path.join(self.provenance_db_path, "runs", f"{run_id}.json")
        if not os.path.exists(run_record_path):
            return {}

        with open(run_record_path, 'r') as f:
            run_record = json.load(f)

        # Get the artifacts
        return run_record.get("stages", {}).get(stage_name, {}).get("artifacts", {})

    def find_reusable_data(self, source_run_id: str = None) -> Dict[str, str]:
        """
        Find data that can be reused from previous runs.

        Args:
            source_run_id: Optional specific run ID to use. If None, will find the best run.

        Returns:
            Dictionary mapping dataset types to paths of reusable data
        """
        result = {"train": "", "val": "", "test": ""}

        # If source run specified, use its data
        if source_run_id:
            run_dir = self.get_run_dir(source_run_id)
            for ds_type in result.keys():
                path = os.path.join(run_dir, "data", f"{ds_type}_data.h5")
                if os.path.exists(path):
                    result[ds_type] = path

            # Check if all datasets found
            if all(result.values()):
                logger.info(f"Using data from specified run {source_run_id}")
                return result
            else:
                logger.warning(f"Not all datasets found in specified run {source_run_id}")
                return {"train": "", "val": "", "test": ""}

        # Find the best run that completed data splitting
        best_run = self.find_best_run_for_stage("data_splitting")
        if not best_run:
            logger.info("No previous run found with completed data splitting")
            return result

        # Get the run ID
        best_run_id = best_run.get("run_id")

        # Check if the datasets exist
        run_dir = self.get_run_dir(best_run_id)
        for ds_type in result.keys():
            path = os.path.join(run_dir, "data", f"{ds_type}_data.h5")
            if os.path.exists(path):
                result[ds_type] = path

        # Only return paths if all datasets are available
        if all(result.values()):
            logger.info(f"Found reusable data from run {best_run_id}")
            return result
        else:
            logger.info("Not all datasets available from best run")
            return {"train": "", "val": "", "test": ""}

    def compare_pipeline_configs(self, pipeline_hash1: str, pipeline_hash2: str) -> Dict[str, Any]:
        """
        Compare two pipeline configurations to identify differences.

        Args:
            pipeline_hash1: Hash of the first pipeline configuration
            pipeline_hash2: Hash of the second pipeline configuration

        Returns:
            Dictionary describing the differences between configurations
        """
        # Find runs with the given hashes
        runs1 = [r for r in self.get_experiment_runs() if r.get("pipeline_hash") == pipeline_hash1]
        runs2 = [r for r in self.get_experiment_runs() if r.get("pipeline_hash") == pipeline_hash2]

        if not runs1 or not runs2:
            return {"error": "One or both pipeline configurations not found"}

        # Use the first run for each hash
        run1 = runs1[0]
        run2 = runs2[0]

        # Extract configurations
        config1 = self._extract_pipeline_config(run1.get("config", {}))
        config2 = self._extract_pipeline_config(run2.get("config", {}))

        # Find differences
        differences = {
            "data": self._find_dict_differences(config1.get("data", {}), config2.get("data", {})),
            "preprocessing": self._find_dict_differences(config1.get("preprocessing", {}),
                                                         config2.get("preprocessing", {})),
            "data_splitting": self._find_dict_differences(config1.get("data_splitting", {}),
                                                          config2.get("data_splitting", {}))
        }

        return {
            "pipeline_hash1": pipeline_hash1,
            "pipeline_hash2": pipeline_hash2,
            "run_id1": run1.get("run_id"),
            "run_id2": run2.get("run_id"),
            "differences": differences
        }

    def _find_dict_differences(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Find differences between two dictionaries."""
        differences = {}

        # Check keys in dict1
        for key in dict1:
            if key not in dict2:
                differences[key] = {"in_first_only": dict1[key]}
            elif dict1[key] != dict2[key]:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    nested_diff = self._find_dict_differences(dict1[key], dict2[key])
                    if nested_diff:
                        differences[key] = nested_diff
                else:
                    differences[key] = {"first": dict1[key], "second": dict2[key]}

        # Check keys in dict2 that aren't in dict1
        for key in dict2:
            if key not in dict1:
                differences[key] = {"in_second_only": dict2[key]}

        return differences

    def save_provenance_graph(self, output_path: Optional[str] = None) -> str:
        """
        Save the complete provenance graph to disk.

        Args:
            output_path: Optional path to save the graph (defaults to provenance DB path)

        Returns:
            Path to the saved graph file
        """
        if not self.provenance_enabled:
            logger.warning("Provenance tracking is disabled, no graph to save")
            return ""

        # If output_path is not provided, use provenance DB path
        if output_path is None:
            output_path = os.path.join(self.provenance_db_path, f"lineage_graph_{self.run_id}.json")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Compile complete provenance graph
        complete_graph = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "end_time": datetime.datetime.now().isoformat(),
            "lineage_graph": self.lineage_graph,
            "data_sources": self.data_sources,
            "transformations": self.transformations,
            "data_splits": self.data_splits,
            "operations": self.operations,
            "stages": self.run_record.get("stages", {})
        }

        # Save to disk
        with open(output_path, 'w') as f:
            json.dump(complete_graph, f, indent=2, default=str)

        logger.info(f"Provenance graph saved to {output_path}")
        return output_path

    def get_data_provenance(self) -> Dict[str, Any]:
        """Get the current data provenance information."""
        if not self.provenance_enabled:
            return {}

        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "data_sources": list(self.data_sources.keys()),
            "transformations": list(self.transformations.keys()),
            "data_splits": list(self.data_splits.keys()),
            "completed_stages": [
                stage for stage, details in self.run_record.get("stages", {}).items()
                if details.get("completed", False)
            ]
        }

    def get_lineage_for_data(self, data_id: str) -> Dict[str, Any]:
        """
        Get the complete lineage for a data entity.

        Args:
            data_id: ID of the data entity

        Returns:
            Complete lineage information for the data entity
        """
        if not self.provenance_enabled or data_id not in self.lineage_graph:
            return {}

        # Start with the data entity
        lineage = {
            "data_id": data_id,
            "ancestors": [],
            "descendants": []
        }

        # Recursively trace ancestors
        self._trace_ancestors(data_id, lineage["ancestors"])

        # Recursively trace descendants
        self._trace_descendants(data_id, lineage["descendants"])

        return lineage

    def _trace_ancestors(self, data_id: str, ancestors_list: List[Dict[str, Any]]) -> None:
        """Recursively trace ancestors of a data entity."""
        if data_id not in self.lineage_graph:
            return

        entity = self.lineage_graph[data_id]

        for input_info in entity.get("inputs", []):
            input_id = input_info.get("id")

            if input_id in self.lineage_graph:
                ancestor_info = {
                    "data_id": input_id,
                    "relationship": input_info,
                    "entity_type": self.lineage_graph[input_id].get("type"),
                    "ancestors": []
                }

                ancestors_list.append(ancestor_info)

                # Recursively trace ancestors of this ancestor
                self._trace_ancestors(input_id, ancestor_info["ancestors"])

    def _trace_descendants(self, data_id: str, descendants_list: List[Dict[str, Any]]) -> None:
        """Recursively trace descendants of a data entity."""
        if data_id not in self.lineage_graph:
            return

        entity = self.lineage_graph[data_id]

        for output_info in entity.get("outputs", []):
            output_id = output_info.get("id")

            if output_id in self.lineage_graph:
                descendant_info = {
                    "data_id": output_id,
                    "relationship": output_info,
                    "entity_type": self.lineage_graph[output_id].get("type"),
                    "descendants": []
                }

                descendants_list.append(descendant_info)

                # Recursively trace descendants of this descendant
                self._trace_descendants(output_id, descendant_info["descendants"])

    def visualize_provenance_graph(self, output_path: Optional[str] = None) -> str:
        """
        Generate a visualization of the provenance graph.

        Args:
            output_path: Optional path to save the visualization

        Returns:
            Path to the saved visualization file
        """
        if not self.provenance_enabled:
            logger.warning("Provenance tracking is disabled, no graph to visualize")
            return ""

        # Check if graphviz is available
        try:
            import graphviz
        except ImportError:
            logger.warning("Graphviz not available, cannot generate visualization")
            return ""

        # If output_path is not provided, use provenance DB path
        if output_path is None:
            output_path = os.path.join(self.provenance_db_path, f"lineage_graph_{self.run_id}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create a directed graph
        dot = graphviz.Digraph(comment='Data Provenance Graph', format='svg')

        # Define node styles
        node_styles = {
            "data_source": {"shape": "cylinder", "color": "blue", "style": "filled", "fillcolor": "lightblue"},
            "transformed_data": {"shape": "cylinder", "color": "green", "style": "filled", "fillcolor": "lightgreen"},
            "train_data": {"shape": "cylinder", "color": "orange", "style": "filled", "fillcolor": "lightsalmon"},
            "validation_data": {"shape": "cylinder", "color": "purple", "style": "filled", "fillcolor": "plum"},
            "test_data": {"shape": "cylinder", "color": "red", "style": "filled", "fillcolor": "lightpink"},
            "trained_model": {"shape": "box", "color": "black", "style": "filled", "fillcolor": "gray90"}
        }

        # Add nodes for all entities in the lineage graph
        for entity_id, entity_info in self.lineage_graph.items():
            entity_type = entity_info.get("type", "unknown")
            label = f"{entity_type}\\n{entity_id[:8]}..."

            # Get node style based on entity type
            style = node_styles.get(entity_type, {"shape": "ellipse"})

            # Add node
            dot.node(entity_id, label, **style)

        # Add edges for relationships
        for entity_id, entity_info in self.lineage_graph.items():
            # Add edges for inputs
            for input_info in entity_info.get("inputs", []):
                input_id = input_info.get("id")
                relationship = next((k for k in input_info.keys() if k.startswith("via_")), None)

                if input_id in self.lineage_graph and relationship:
                    # Label the edge with the relationship type
                    label = relationship.replace("via_", "")
                    dot.edge(input_id, entity_id, label=label)

        # Render the graph
        dot.render(output_path, cleanup=True)

        logger.info(f"Provenance graph visualization saved to {output_path}.svg")
        return f"{output_path}.svg"

    @staticmethod
    def list_experiments(base_output_dir: str = "./output") -> List[str]:
        """
        List all experiments in the base output directory.

        Args:
            base_output_dir: Base output directory

        Returns:
            List of experiment names
        """
        if not os.path.exists(base_output_dir):
            return []

        experiments = []
        for dirname in os.listdir(base_output_dir):
            exp_dir = os.path.join(base_output_dir, dirname)
            if os.path.isdir(exp_dir) and os.path.exists(os.path.join(exp_dir, "provenance")):
                experiments.append(dirname)

        return sorted(experiments)
