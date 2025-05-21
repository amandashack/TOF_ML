#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database API for the ML Provenance Tracker Framework.
This module handles interactions with the experiment tracking database.
"""

import os
import logging
import datetime
import json
import sqlite3
import yaml
from typing import Dict, Any, Optional, List, Union, Tuple

logger = logging.getLogger(__name__)

class DBApi:
    """
    API for interacting with the experiment tracking database.
    
    This class provides methods to:
    - Record experiment runs
    - Query experiment results
    - Track model artifacts
    - Compare experiment results
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the database API.
        
        Args:
            config_path: Path to database configuration file
        """
        self.config = self._load_config(config_path)
        self.db_type = self.config.get("type", "sqlite")
        self.connection = None
        
        # Connect to the database
        self._connect_to_database()
        
        # Initialize database schema if needed
        self._initialize_database()
        
        logger.info(f"Database API initialized with {self.db_type} backend")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load database configuration from YAML file."""
        if not os.path.exists(config_path):
            logger.warning(f"Database config file not found: {config_path}")
            return {"type": "sqlite", "path": ":memory:"}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _connect_to_database(self):
        """Connect to the database based on configuration."""
        if self.db_type == "sqlite":
            db_path = self.config.get("path", ":memory:")
            
            # Create directory if it doesn't exist
            if db_path != ":memory:":
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            self.connection = sqlite3.connect(db_path)
            self.connection.row_factory = sqlite3.Row
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _initialize_database(self):
        """Initialize database schema if it doesn't exist."""
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()
            
            # Create experiments table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT,
                timestamp TEXT,
                config TEXT,
                metadata TEXT
            )
            ''')
            
            # Create models table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                experiment_id TEXT,
                path TEXT,
                metadata TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
            ''')
            
            # Create metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id TEXT PRIMARY KEY,
                experiment_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_type TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
            ''')
            
            # Create artifacts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                experiment_id TEXT,
                artifact_type TEXT,
                path TEXT,
                metadata TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
            ''')
            
            self.connection.commit()
    
    def record_experiment(
        self,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
        training_results: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        report_paths: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Record an experiment run in the database.
        
        Args:
            metadata: Experiment metadata
            config: Experiment configuration
            training_results: Optional training results
            evaluation_results: Optional evaluation results
            model_path: Optional path to saved model
            report_paths: Optional paths to generated reports
            
        Returns:
            Experiment ID
        """
        logger.info("Recording experiment in database")
        
        # Generate experiment ID if not provided
        experiment_id = metadata.get("experiment_id", str(datetime.datetime.now().isoformat()))
        
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()
            
            # Insert experiment record
            cursor.execute(
                "INSERT INTO experiments (id, name, timestamp, config, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    metadata.get("experiment_name", "unnamed_experiment"),
                    datetime.datetime.now().isoformat(),
                    json.dumps(config),
                    json.dumps(metadata)
                )
            )
            
            # Record metrics
            self._record_metrics(cursor, experiment_id, training_results, evaluation_results)
            
            # Record model if available
            if model_path:
                self._record_model(cursor, experiment_id, model_path, metadata)
            
            # Record report artifacts if available
            if report_paths:
                self._record_artifacts(cursor, experiment_id, report_paths)
            
            self.connection.commit()
        
        logger.info(f"Experiment recorded with ID: {experiment_id}")
        return experiment_id
    
    def _record_metrics(
        self,
        cursor,
        experiment_id: str,
        training_results: Optional[Dict[str, Any]],
        evaluation_results: Optional[Dict[str, Any]]
    ):
        """Record metrics in the database."""
        # Record training metrics
        if training_results:
            for key, value in training_results.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_id = f"{experiment_id}_{key}"
                    cursor.execute(
                        "INSERT INTO metrics (id, experiment_id, metric_name, metric_value, metric_type) VALUES (?, ?, ?, ?, ?)",
                        (
                            metric_id,
                            experiment_id,
                            key,
                            float(value),
                            "training"
                        )
                    )
        
        # Record evaluation metrics
        if evaluation_results:
            for key, value in evaluation_results.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in ["y_true", "y_pred"]:
                    metric_id = f"{experiment_id}_{key}"
                    cursor.execute(
                        "INSERT INTO metrics (id, experiment_id, metric_name, metric_value, metric_type) VALUES (?, ?, ?, ?, ?)",
                        (
                            metric_id,
                            experiment_id,
                            key,
                            float(value),
                            "evaluation"
                        )
                    )
    
    def _record_model(
        self,
        cursor,
        experiment_id: str,
        model_path: str,
        metadata: Dict[str, Any]
    ):
        """Record model in the database."""
        model_id = f"{experiment_id}_model"
        cursor.execute(
            "INSERT INTO models (id, experiment_id, path, metadata) VALUES (?, ?, ?, ?)",
            (
                model_id,
                experiment_id,
                model_path,
                json.dumps(metadata)
            )
        )
    
    def _record_artifacts(
        self,
        cursor,
        experiment_id: str,
        artifact_paths: Dict[str, str]
    ):
        """Record artifacts in the database."""
        for artifact_type, path in artifact_paths.items():
            if path:
                artifact_id = f"{experiment_id}_{artifact_type}"
                cursor.execute(
                    "INSERT INTO artifacts (id, experiment_id, artifact_type, path, metadata) VALUES (?, ?, ?, ?, ?)",
                    (
                        artifact_id,
                        experiment_id,
                        artifact_type,
                        path,
                        json.dumps({"timestamp": datetime.datetime.now().isoformat()})
                    )
                )

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment details from the database.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Experiment details
        """
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()

            # Get experiment record
            cursor.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,)
            )
            experiment_row = cursor.fetchone()

            if not experiment_row:
                logger.warning(f"Experiment not found: {experiment_id}")
                return {}

            # Convert row to dictionary
            experiment = dict(experiment_row)

            # Parse JSON fields
            experiment["config"] = json.loads(experiment["config"])
            experiment["metadata"] = json.loads(experiment["metadata"])

            # Get metrics
            cursor.execute(
                "SELECT metric_name, metric_value, metric_type FROM metrics WHERE experiment_id = ?",
                (experiment_id,)
            )
            metrics_rows = cursor.fetchall()

            experiment["metrics"] = {row["metric_name"]: row["metric_value"] for row in metrics_rows}

            # Get model information
            cursor.execute(
                "SELECT * FROM models WHERE experiment_id = ?",
                (experiment_id,)
            )
            model_row = cursor.fetchone()

            if model_row:
                experiment["model"] = dict(model_row)
                experiment["model"]["metadata"] = json.loads(experiment["model"]["metadata"])

            # Get artifacts
            cursor.execute(
                "SELECT artifact_type, path FROM artifacts WHERE experiment_id = ?",
                (experiment_id,)
            )
            artifact_rows = cursor.fetchall()

            experiment["artifacts"] = {row["artifact_type"]: row["path"] for row in artifact_rows}

            return experiment

        return {}

    def query_experiments(
            self,
            filters: Optional[Dict[str, Any]] = None,
            sort_by: Optional[str] = None,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query experiments from the database.

        Args:
            filters: Optional filters to apply
            sort_by: Optional metric to sort by
            limit: Maximum number of results to return

        Returns:
            List of matching experiments
        """
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()

            # Base query
            query = "SELECT id, name, timestamp FROM experiments"
            params = []

            # Apply filters if provided
            if filters:
                filter_clauses = []

                # Filter by name
                if "name" in filters:
                    filter_clauses.append("name LIKE ?")
                    params.append(f"%{filters['name']}%")

                # Filter by timestamp
                if "timestamp_start" in filters:
                    filter_clauses.append("timestamp >= ?")
                    params.append(filters["timestamp_start"])

                if "timestamp_end" in filters:
                    filter_clauses.append("timestamp <= ?")
                    params.append(filters["timestamp_end"])

                # Filter by model type or other metadata fields
                if "model_type" in filters:
                    filter_clauses.append("json_extract(metadata, '$.model_type') = ?")
                    params.append(filters["model_type"])

                # Add WHERE clause if there are filters
                if filter_clauses:
                    query += " WHERE " + " AND ".join(filter_clauses)

            # Apply sorting if provided
            if sort_by:
                # Sort by timestamp by default
                if sort_by == "timestamp":
                    query += " ORDER BY timestamp DESC"
                # Sort by a metric if specified
                else:
                    # Join with metrics table to sort by a specific metric
                    query = f"""
                    SELECT e.id, e.name, e.timestamp, m.metric_value
                    FROM experiments e
                    LEFT JOIN metrics m ON e.id = m.experiment_id AND m.metric_name = ?
                    """
                    params.insert(0, sort_by)

                    # Re-apply filters if any
                    if filters and len(filter_clauses) > 0:
                        query += " WHERE " + " AND ".join(filter_clauses)

                    # Add sorting direction
                    query += " ORDER BY m.metric_value ASC"
            else:
                # Default sorting by timestamp
                query += " ORDER BY timestamp DESC"

            # Apply limit
            query += " LIMIT ?"
            params.append(limit)

            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert rows to dictionaries
            experiments = [dict(row) for row in rows]

            return experiments

        return []
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Comparison results
        """
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()
            
            # Get basic experiment information
            experiments = []
            for exp_id in experiment_ids:
                cursor.execute(
                    "SELECT id, name, timestamp FROM experiments WHERE id = ?",
                    (exp_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    experiments.append(dict(row))
            
            # Get metrics for all experiments
            metrics_comparison = {}
            for exp_id in experiment_ids:
                cursor.execute(
                    "SELECT metric_name, metric_value, metric_type FROM metrics WHERE experiment_id = ?",
                    (exp_id,)
                )
                rows = cursor.fetchall()
                
                exp_metrics = {}
                for row in rows:
                    metric_name = row["metric_name"]
                    metric_value = row["metric_value"]
                    metric_type = row["metric_type"]
                    
                    if metric_name not in metrics_comparison:
                        metrics_comparison[metric_name] = {"type": metric_type, "values": {}}
                    
                    metrics_comparison[metric_name]["values"][exp_id] = metric_value
            
            return {
                "experiments": experiments,
                "metrics": metrics_comparison
            }
        
        return {}

    def get_metrics(self, experiment_id: str) -> Dict[str, float]:
        """
        Get all metrics for an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Dictionary of metrics
        """
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()

            cursor.execute(
                "SELECT metric_name, metric_value FROM metrics WHERE experiment_id = ?",
                (experiment_id,)
            )
            rows = cursor.fetchall()

            return {row["metric_name"]: row["metric_value"] for row in rows}

        return {}

    def get_best_experiment(self, metric_name: str, higher_is_better: bool = False) -> Dict[str, Any]:
        """
        Get the experiment with the best metric value.

        Args:
            metric_name: Name of the metric to compare
            higher_is_better: Whether higher values are better

        Returns:
            Best experiment details
        """
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()

            # Order by metric value (ascending or descending)
            order = "DESC" if higher_is_better else "ASC"

            # Get experiment with the best metric
            query = f"""
            SELECT e.id
            FROM experiments e
            JOIN metrics m ON e.id = m.experiment_id
            WHERE m.metric_name = ?
            ORDER BY m.metric_value {order}
            LIMIT 1
            """

            cursor.execute(query, (metric_name,))
            row = cursor.fetchone()

            if row:
                return self.get_experiment(row["id"])

        return {}
    
    def get_best_model(self, metric_name: str, is_higher_better: bool = False) -> Dict[str, Any]:
        """
        Get the best model based on a specific metric.
        
        Args:
            metric_name: Metric to use for comparison
            is_higher_better: Whether higher values are better
            
        Returns:
            Best model information
        """
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()
            
            # Order by metric value (ascending or descending)
            order = "DESC" if is_higher_better else "ASC"
            
            # Get experiment with the best metric
            query = f"""
            SELECT e.id, e.name, e.timestamp, m.metric_value
            FROM experiments e
            JOIN metrics m ON e.id = m.experiment_id
            WHERE m.metric_name = ?
            ORDER BY m.metric_value {order}
            LIMIT 1
            """
            
            cursor.execute(query, (metric_name,))
            exp_row = cursor.fetchone()
            
            if not exp_row:
                logger.warning(f"No experiments found with metric: {metric_name}")
                return {}
            
            # Get the model for this experiment
            exp_id = exp_row["id"]
            cursor.execute(
                "SELECT * FROM models WHERE experiment_id = ?",
                (exp_id,)
            )
            model_row = cursor.fetchone()
            
            if not model_row:
                logger.warning(f"No model found for best experiment: {exp_id}")
                return {"experiment": dict(exp_row), "model": None}
            
            return {
                "experiment": dict(exp_row),
                "model": dict(model_row)
            }
        
        return {}
    
    def record_model_run(
        self,
        config_dict: Dict[str, Any],
        training_results: Dict[str, Any],
        model_path: str,
        plot_paths: Dict[str, str]
    ) -> None:
        """
        Record a model training run.
        
        This method is a convenience wrapper for record_experiment.
        
        Args:
            config_dict: Configuration dictionary
            training_results: Training results
            model_path: Path to saved model
            plot_paths: Paths to generated plots
        """
        # Create metadata from config and results
        metadata = {
            "experiment_name": config_dict.get("experiment_name", "unnamed_run"),
            "timestamp": datetime.datetime.now().isoformat(),
            "training_results": {k: v for k, v in training_results.items() if isinstance(v, (int, float))}
        }
        
        # Record as an experiment
        self.record_experiment(
            metadata=metadata,
            config=config_dict,
            training_results=training_results,
            model_path=model_path,
            report_paths=plot_paths
        )
        
        logger.info("Model run recorded successfully")

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment and all associated data from the database.

        Args:
            experiment_id: ID of the experiment to delete

        Returns:
            True if successful, False otherwise
        """
        if self.db_type == "sqlite":
            try:
                cursor = self.connection.cursor()

                # Delete related records first
                cursor.execute("DELETE FROM metrics WHERE experiment_id = ?", (experiment_id,))
                cursor.execute("DELETE FROM models WHERE experiment_id = ?", (experiment_id,))
                cursor.execute("DELETE FROM artifacts WHERE experiment_id = ?", (experiment_id,))

                # Delete the experiment record
                cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))

                # Commit changes
                self.connection.commit()

                return True
            except Exception as e:
                logger.error(f"Error deleting experiment: {e}")
                self.connection.rollback()
                return False

        return False
    
    def __del__(self):
        """Clean up database connection on object destruction."""
        if self.connection:
            self.connection.close()
