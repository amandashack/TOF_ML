#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQLite implementations of repository and query service interfaces.
"""

import os
import json
import datetime
import logging
from typing import Dict, Any, List, Optional, Tuple

from .base_repository import BaseRepository, BaseQueryService

logger = logging.getLogger(__name__)


class SQLiteRepository(BaseRepository):
    """SQLite implementation of repository pattern (write operations)."""

    def __init__(self, connection, api_instance):
        self.connection = connection
        self.api = api_instance  # Reference to main API for utility methods

    def create_experiment(self, name: str, config: Dict[str, Any], description: str = "") -> str:
        """Create experiment or return existing one with same name."""
        # Generate stable experiment ID
        experiment_id = self.api.generate_experiment_id(name)

        cursor = self.connection.cursor()

        # Check if experiment already exists
        cursor.execute("SELECT id, created_at FROM experiments WHERE id = ?", (experiment_id,))
        existing = cursor.fetchone()

        if existing:
            logger.info(f"Using existing experiment: {experiment_id} (created: {existing['created_at']})")
            return experiment_id

        # Create new experiment
        git_commit = self.api.get_git_commit_sha()
        pipeline_hash = self.api._generate_pipeline_hash(config)

        cursor.execute(
            "INSERT INTO experiments (id, name, created_at, description, git_commit_sha, pipeline_hash) VALUES (?, ?, ?, ?, ?, ?)",
            (
                experiment_id,
                name,
                datetime.datetime.now().isoformat(),
                description,
                git_commit,
                pipeline_hash
            )
        )
        self.connection.commit()

        logger.info(f"Created new experiment: {experiment_id}")
        return experiment_id

    def create_run(self, experiment_id: str, config: Dict[str, Any], force_reload: bool = False) -> str:
        """Create a new run."""
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        pipeline_hash = self.api._generate_pipeline_hash(config)
        environment_hash = self.api.get_environment_hash()

        # Check if data can be reused
        data_reused, source_run_id = self.api._check_data_reuse(experiment_id, pipeline_hash, force_reload)

        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO runs (id, experiment_id, created_at, pipeline_hash, data_reused, force_reload, source_run_id, environment_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                experiment_id,
                datetime.datetime.now().isoformat(),
                pipeline_hash,
                data_reused,
                force_reload,
                source_run_id,
                environment_hash
            )
        )
        self.connection.commit()

        logger.info(f"Created run: {run_id} (data_reused: {data_reused})")
        return run_id

    def complete_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a run as completed."""
        cursor = self.connection.cursor()
        cursor.execute(
            "UPDATE runs SET completed_at = ?, status = ? WHERE id = ?",
            (datetime.datetime.now().isoformat(), status, run_id)
        )
        self.connection.commit()

    def store_artifact(self, run_id: str, file_path: str, artifact_type: str,
                       artifact_role: str, stage: str, metadata: Dict[str, Any] = None) -> str:
        """Store artifact with minimal metadata."""
        # Calculate hash
        hash_id = self.api._calculate_file_hash(file_path)

        # Get file size
        size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        # Create simple metadata string instead of JSON
        metadata_str = ""
        if metadata:
            # Convert to simple key=value pairs
            metadata_parts = []
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata_parts.append(f"{key}={value}")
                elif isinstance(value, dict):
                    # Skip complex nested data
                    metadata_parts.append(f"{key}=<dict>")
                else:
                    metadata_parts.append(f"{key}={type(value).__name__}")
            metadata_str = "; ".join(metadata_parts)

        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO artifacts (hash_id, run_id, artifact_role, stage, artifact_type, file_path, metadata, created_at, size_bytes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                hash_id,
                run_id,
                artifact_role,
                stage,
                artifact_type,
                file_path,
                metadata_str,  # Simple string instead of JSON
                datetime.datetime.now().isoformat(),
                size_bytes
            )
        )
        self.connection.commit()

        return hash_id

    def store_metrics(self, run_id: str, metrics: Dict[str, Any], stage: str,
                      epoch: int = None, metric_type: str = "predefined",
                      metric_category: str = "custom") -> None:
        """Store metrics with enhanced categorization."""
        cursor = self.connection.cursor()

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                cursor.execute(
                    "INSERT INTO metrics (run_id, metric_name, metric_value, metric_type, metric_category, stage, epoch, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        metric_name,
                        float(metric_value),
                        metric_type,
                        metric_category,
                        stage,
                        epoch,
                        datetime.datetime.now().isoformat()
                    )
                )

        self.connection.commit()

    def store_parameters(self, run_id: str, parameters: Dict[str, Any], stage: str) -> None:
        """Store parameters for a run stage."""
        cursor = self.connection.cursor()

        for param_name, param_value in parameters.items():
            param_type = type(param_value).__name__
            cursor.execute(
                "INSERT OR REPLACE INTO parameters (run_id, parameter_name, parameter_value, parameter_type, stage) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    param_name,
                    json.dumps(param_value) if isinstance(param_value, (dict, list)) else str(param_value),
                    param_type,
                    stage
                )
            )

        self.connection.commit()

    def record_data_transformation(self, run_id: str, input_hash: str, output_hash: str,
                                   transformation_type: str, transformation_config: Dict[str, Any],
                                   stage: str) -> None:
        """Record a data transformation in lineage."""
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO data_lineage (run_id, input_hash, output_hash, transformation_name, stage, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                input_hash,
                output_hash,
                transformation_type,
                stage,
                datetime.datetime.now().isoformat()
            )
        )
        self.connection.commit()


class SQLiteQueryService(BaseQueryService):
    """SQLite implementation of query service (read operations)."""

    def __init__(self, connection, api_instance):
        self.connection = connection
        self.api = api_instance  # Reference to main API for utility methods

    def find_experiment_by_name_or_id(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Find experiment by name, ID, or partial match."""
        cursor = self.connection.cursor()

        # Try exact ID match first
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (identifier,))
        result = cursor.fetchone()
        if result:
            return dict(result)

        # Try exact name match
        cursor.execute("SELECT * FROM experiments WHERE name = ?", (identifier,))
        result = cursor.fetchone()
        if result:
            return dict(result)

        # Try to generate ID from name and match
        try:
            generated_id = self.api.generate_experiment_id(identifier)
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (generated_id,))
            result = cursor.fetchone()
            if result:
                return dict(result)
        except:
            pass

        # Try partial matches on name or ID
        cursor.execute(
            "SELECT * FROM experiments WHERE name LIKE ? OR id LIKE ? LIMIT 1",
            (f"%{identifier}%", f"%{identifier}%")
        )
        result = cursor.fetchone()
        if result:
            return dict(result)

        return None

    def get_experiments(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of experiments with run counts."""
        cursor = self.connection.cursor()

        query = """
        SELECT e.id, e.name, e.created_at, e.git_commit_sha,
               COUNT(r.id) as total_runs,
               COUNT(CASE WHEN r.status = 'completed' THEN 1 END) as completed_runs
        FROM experiments e
        LEFT JOIN runs r ON e.id = r.experiment_id
        GROUP BY e.id, e.name, e.created_at, e.git_commit_sha
        ORDER BY e.created_at DESC
        LIMIT ?
        """

        cursor.execute(query, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_runs_for_experiment(self, experiment_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get runs for a specific experiment."""
        cursor = self.connection.cursor()

        query = """
        SELECT r.id, r.created_at, r.completed_at, r.status, r.data_reused,
               AVG(CASE WHEN m.metric_name = 'test_loss' THEN m.metric_value END) as avg_test_loss
        FROM runs r
        LEFT JOIN metrics m ON r.id = m.run_id
        WHERE r.experiment_id = ?
        GROUP BY r.id, r.created_at, r.completed_at, r.status, r.data_reused
        ORDER BY r.created_at DESC
        LIMIT ?
        """

        cursor.execute(query, (experiment_id, limit))
        return [dict(row) for row in cursor.fetchall()]

    def get_run_lineage(self, run_id: str) -> Dict[str, Any]:
        """Get enhanced complete lineage for a run."""
        cursor = self.connection.cursor()

        # Get run info
        cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        run_info = dict(cursor.fetchone() or {})

        # Get artifacts
        cursor.execute("SELECT * FROM artifacts WHERE run_id = ?", (run_id,))
        artifacts = [dict(row) for row in cursor.fetchall()]

        # Get transformation lineage
        cursor.execute("""
            SELECT dl.*
            FROM data_lineage dl
            WHERE dl.run_id = ?
            ORDER BY dl.created_at
        """, (run_id,))
        transformation_lineage = [dict(row) for row in cursor.fetchall()]

        # Get parameters
        cursor.execute("SELECT * FROM parameters WHERE run_id = ?", (run_id,))
        parameters = [dict(row) for row in cursor.fetchall()]

        # Get metrics
        cursor.execute("SELECT * FROM metrics WHERE run_id = ? ORDER BY created_at", (run_id,))
        metrics = [dict(row) for row in cursor.fetchall()]

        return {
            "run_info": run_info,
            "artifacts": artifacts,
            "transformation_lineage": transformation_lineage,
            "parameters": parameters,
            "metrics": metrics
        }

    def get_reusable_artifacts(self, experiment_id: str, pipeline_hash: str) -> Dict[str, str]:
        """Get reusable artifacts from previous runs."""
        cursor = self.connection.cursor()

        logger.info(f"Looking for reusable artifacts: exp={experiment_id}, hash={pipeline_hash}")

        # Find artifacts from data_splitting stage
        query = """
            SELECT r.id, a.hash_id, a.artifact_role, a.file_path, a.stage
            FROM runs r
            JOIN artifacts a ON r.id = a.run_id
            WHERE r.experiment_id = ? AND r.pipeline_hash = ? 
                AND a.artifact_type = 'data' AND r.status = 'completed'
                AND a.artifact_role IN ('train_data', 'val_data', 'test_data')
            ORDER BY r.created_at DESC
            LIMIT 10
        """

        cursor.execute(query, (experiment_id, pipeline_hash))
        results = cursor.fetchall()

        logger.info(f"Found {len(results)} potential reusable data artifacts")

        if not results:
            # Debug: Check what experiments and pipeline hashes exist
            cursor.execute("SELECT DISTINCT experiment_id, pipeline_hash FROM runs WHERE status = 'completed'")
            all_runs = cursor.fetchall()
            logger.info(f"Available completed runs: {[dict(row) for row in all_runs]}")
            return {}

        # Group by run and get the most recent complete set
        artifacts_by_run = {}
        for row in results:
            run_id = row["id"]
            if run_id not in artifacts_by_run:
                artifacts_by_run[run_id] = {}

            if os.path.exists(row["file_path"]):
                artifacts_by_run[run_id][row["artifact_role"]] = row["hash_id"]
                logger.info(f"Found reusable artifact: {row['artifact_role']} -> {row['hash_id']}")
            else:
                logger.warning(f"Artifact file missing: {row['file_path']}")

        # Find a run with all three required artifacts
        required_roles = {'train_data', 'val_data', 'test_data'}
        for run_id, artifacts in artifacts_by_run.items():
            if required_roles.issubset(set(artifacts.keys())):
                logger.info(f"Found complete set of reusable artifacts from run {run_id}")
                return artifacts

        logger.info("No complete set of reusable artifacts found")
        return {}

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        comparison = {"runs": [], "metrics_comparison": {}, "parameter_comparison": {}}

        cursor = self.connection.cursor()

        for run_id in run_ids:
            # Get run info
            cursor.execute("""
                SELECT r.*, e.name as experiment_name
                FROM runs r
                JOIN experiments e ON r.experiment_id = e.id
                WHERE r.id = ?
            """, (run_id,))
            run_info = dict(cursor.fetchone() or {})
            comparison["runs"].append(run_info)

            # Get metrics for comparison
            cursor.execute("SELECT metric_name, metric_value, stage FROM metrics WHERE run_id = ?", (run_id,))
            metrics = cursor.fetchall()

            for metric in metrics:
                key = f"{metric['stage']}_{metric['metric_name']}"
                if key not in comparison["metrics_comparison"]:
                    comparison["metrics_comparison"][key] = {}
                comparison["metrics_comparison"][key][run_id] = metric["metric_value"]

            # Get parameters for comparison
            cursor.execute("SELECT parameter_name, parameter_value, stage FROM parameters WHERE run_id = ?", (run_id,))
            parameters = cursor.fetchall()

            for param in parameters:
                key = f"{param['stage']}_{param['parameter_name']}"
                if key not in comparison["parameter_comparison"]:
                    comparison["parameter_comparison"][key] = {}
                comparison["parameter_comparison"][key][run_id] = param["parameter_value"]

        return comparison

    def query_artifacts_by_criteria(self, artifact_type: str = None, stage: str = None,
                                    limit: int = 20) -> List[Dict[str, Any]]:
        """Query artifacts by type or stage."""
        cursor = self.connection.cursor()

        query = """
        SELECT a.hash_id, a.artifact_type, a.file_path, a.created_at, a.size_bytes,
               a.artifact_role, a.stage, a.run_id
        FROM artifacts a
        WHERE 1=1
        """
        params = []

        if artifact_type:
            query += " AND a.artifact_type = ?"
            params.append(artifact_type)

        if stage:
            query += " AND a.stage = ?"
            params.append(stage)

        query += " ORDER BY a.created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_database_summary(self) -> Dict[str, Any]:
        """Get database summary statistics."""
        cursor = self.connection.cursor()

        # Get counts
        cursor.execute("SELECT COUNT(*) as count FROM experiments")
        exp_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM runs")
        run_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM runs WHERE status = 'completed'")
        completed_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM artifacts")
        artifact_count = cursor.fetchone()["count"]

        cursor.execute("SELECT SUM(size_bytes) as total_size FROM artifacts")
        total_size = cursor.fetchone()["total_size"] or 0

        cursor.execute("SELECT COUNT(*) as count FROM runs WHERE data_reused = 1")
        reused_count = cursor.fetchone()["count"]

        return {
            "experiments": exp_count,
            "runs": run_count,
            "completed_runs": completed_count,
            "failed_runs": run_count - completed_count,
            "artifacts": artifact_count,
            "total_storage_bytes": total_size,
            "data_reused_runs": reused_count,
            "success_rate": (completed_count / run_count) * 100 if run_count > 0 else 0,
            "data_reuse_rate": (reused_count / run_count) * 100 if run_count > 0 else 0
        }

    def delete_run_cascade(self, run_id: str) -> bool:
        """Delete a run and all associated data."""
        try:
            cursor = self.connection.cursor()

            # Delete in correct order to respect foreign key constraints
            cursor.execute(
                "DELETE FROM transformation_parameters WHERE lineage_id IN (SELECT id FROM data_lineage WHERE run_id = ?)",
                (run_id,))
            cursor.execute(
                "DELETE FROM model_parameters WHERE model_lineage_id IN (SELECT id FROM model_lineage WHERE run_id = ?)",
                (run_id,))
            cursor.execute("DELETE FROM data_lineage WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM model_lineage WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM parameters WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM runs WHERE id = ?", (run_id,))

            self.connection.commit()
            return True

        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error deleting run {run_id}: {e}")
            return False

    def get_training_progress(self, run_id: str) -> Dict[str, Any]:
        """Get real-time training progress for a run."""
        cursor = self.connection.cursor()

        # Get latest training metrics
        cursor.execute("""
            SELECT epoch, metric_name, metric_value, created_at
            FROM metrics 
            WHERE run_id = ? AND stage = 'training'
            ORDER BY epoch DESC, created_at DESC
        """, (run_id,))

        metrics = cursor.fetchall()

        if not metrics:
            return {"status": "no_training_data"}

        # Get latest epoch metrics
        latest_epoch = metrics[0]["epoch"]
        latest_metrics = {m["metric_name"]: m["metric_value"]
                          for m in metrics if m["epoch"] == latest_epoch}

        # Get best metrics so far
        cursor.execute("""
            SELECT MIN(metric_value) as best_val_loss
            FROM metrics 
            WHERE run_id = ? AND metric_name = 'val_loss'
        """, (run_id,))

        best_result = cursor.fetchone()
        best_val_loss = best_result["best_val_loss"] if best_result else None

        # Get training history
        cursor.execute("""
            SELECT epoch, 
                   AVG(CASE WHEN metric_name = 'loss' THEN metric_value END) as train_loss,
                   AVG(CASE WHEN metric_name = 'val_loss' THEN metric_value END) as val_loss,
                   AVG(CASE WHEN metric_name = 'accuracy' THEN metric_value END) as train_acc,
                   AVG(CASE WHEN metric_name = 'val_accuracy' THEN metric_value END) as val_acc
            FROM metrics 
            WHERE run_id = ? AND stage = 'training' AND metric_name IN ('loss', 'val_loss', 'accuracy', 'val_accuracy')
            GROUP BY epoch
            ORDER BY epoch
        """, (run_id,))

        history = [dict(row) for row in cursor.fetchall()]

        return {
            "status": "training",
            "current_epoch": latest_epoch,
            "latest_metrics": latest_metrics,
            "best_val_loss": best_val_loss,
            "training_history": history,
            "total_epochs": len(history)
        }

    def check_data_reuse(self, experiment_id: str, pipeline_hash: str, force_reload: bool) -> Tuple[
        bool, Optional[str]]:
        """Check if data can be reused from previous runs."""
        if force_reload:
            return False, None

        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id FROM runs 
            WHERE experiment_id = ? AND pipeline_hash = ? AND status = 'completed'
            AND EXISTS (
                SELECT 1 FROM artifacts ra 
                WHERE ra.run_id = runs.id AND ra.stage = 'data_splitting'
            )
            ORDER BY created_at DESC
            LIMIT 1
        """, (experiment_id, pipeline_hash))

        result = cursor.fetchone()
        if result:
            return True, result["id"]

        return False, None

    def get_environment_info(self, env_hash: str) -> Optional[Dict[str, Any]]:
        """Get environment information by hash."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM environments WHERE hash_id = ?", (env_hash,))
        result = cursor.fetchone()
        return dict(result) if result else None

    def store_environment(self, env_hash: str, python_version: str, packages: str, system_info: str) -> None:
        """Store environment information."""
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO environments (hash_id, python_version, packages, system_info, created_at) VALUES (?, ?, ?, ?, ?)",
            (env_hash, python_version, packages, system_info, datetime.datetime.now().isoformat())
        )
        self.connection.commit()

    def get_recent_runs_for_experiment(self, experiment_id: str, current_run_id: str, limit: int = 5) -> List[
        Dict[str, Any]]:
        """Get recent runs for an experiment excluding current run."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT id FROM runs WHERE experiment_id = ? AND id != ? ORDER BY created_at DESC LIMIT ?",
            (experiment_id, current_run_id, limit)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_run_status(self, run_id: str) -> Optional[str]:
        """Get the status of a run."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT status FROM runs WHERE id = ?", (run_id,))
        result = cursor.fetchone()
        return result["status"] if result else None