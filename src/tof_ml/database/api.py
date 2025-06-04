#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Database API for the ML Provenance Tracker Framework.
This module handles interactions with the experiment tracking database using repository pattern.
"""

import os
import logging
import datetime
import json
import sqlite3
import yaml
import hashlib
import re
from typing import Dict, Any, Optional, List, Tuple
import subprocess

from src.tof_ml.database.base_repository import BaseRepository, BaseQueryService
from src.tof_ml.database.sqlite import SQLiteRepository, SQLiteQueryService

logger = logging.getLogger(__name__)


class DBApi:
    """
    Enhanced database API that composes repository and query services.
    Provides database abstraction through repository pattern.
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

        # Initialize repository and query services
        if self.db_type == "sqlite":
            self.repository = SQLiteRepository(self.connection, self)
            self.query_service = SQLiteQueryService(self.connection, self)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

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
        """Initialize the updated database schema."""
        if self.db_type == "sqlite":
            cursor = self.connection.cursor()

            # Create experiments table with unique constraint on name
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                description TEXT,
                git_commit_sha TEXT,
                pipeline_hash TEXT NOT NULL
            )
            ''')

            # Create runs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT DEFAULT 'running',
                pipeline_hash TEXT NOT NULL,
                data_reused BOOLEAN DEFAULT FALSE,
                force_reload BOOLEAN DEFAULT FALSE,
                source_run_id TEXT,
                environment_hash TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                FOREIGN KEY (source_run_id) REFERENCES runs(id),
                FOREIGN KEY (environment_hash) REFERENCES environments(hash_id)
            )
            ''')

            # Create combined artifacts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                artifact_role TEXT NOT NULL,
                stage TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                size_bytes INTEGER,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            ''')

            # Create enhanced data lineage table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                input_hash TEXT,
                output_hash TEXT NOT NULL,
                transformation_name TEXT NOT NULL,
                stage TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            ''')

            # Create enhanced metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT NOT NULL,
                metric_category TEXT NOT NULL,
                stage TEXT NOT NULL,
                epoch INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            ''')

            # Create parameters table (for run/stage parameters)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                parameter_name TEXT NOT NULL,
                parameter_value TEXT NOT NULL,
                parameter_type TEXT NOT NULL,
                stage TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            ''')

            # Create environment table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS environments (
                hash_id TEXT PRIMARY KEY,
                python_version TEXT,
                packages TEXT,
                system_info TEXT,
                created_at TEXT NOT NULL
            )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(hash_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lineage_run ON data_lineage(run_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)')

            self.connection.commit()

    @staticmethod
    def slugify(name: str) -> str:
        """Convert a name to a URL-friendly slug."""
        name = name.lower().strip()
        name = re.sub(r'[^a-z0-9]+', '-', name)
        return name.strip('-')

    @staticmethod
    def generate_experiment_id(name: str) -> str:
        """Generate a stable experiment ID from name using slug + hash."""
        slug = DBApi.slugify(name)
        hash_suffix = hashlib.sha256(name.encode('utf-8')).hexdigest()[:6]
        return f"{slug}-{hash_suffix}"

    def get_git_commit_sha(self) -> str:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                    capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not retrieve git commit SHA")
            return "unknown"

    def get_environment_hash(self) -> str:
        """Get environment hash for reproducibility."""
        import platform
        import sys
        import pkg_resources

        # Get system information
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()
        }

        # Get installed packages
        packages = {}
        for dist in pkg_resources.working_set:
            packages[dist.project_name] = dist.version

        # Create hash
        env_data = json.dumps({
            "system_info": system_info,
            "packages": packages
        }, sort_keys=True)

        env_hash = hashlib.sha256(env_data.encode()).hexdigest()

        # Store environment if not exists
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO environments (hash_id, python_version, packages, system_info, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                env_hash,
                sys.version,
                json.dumps(packages),
                json.dumps(system_info),
                datetime.datetime.now().isoformat()
            )
        )
        self.connection.commit()

        return env_hash

    def _generate_pipeline_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash for pipeline configuration."""
        # Extract relevant config parts
        pipeline_config = {
            "data": config.get("data", {}),
            "model": config.get("model", {}),
            "plugins": config.get("plugins", {}),
            "transformations": config.get("transformations", {})
        }

        # Remove fields that don't affect processing
        pipeline_config["data"].pop("force_reload", None)

        config_str = json.dumps(pipeline_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file or directory."""
        if not os.path.exists(file_path):
            return hashlib.sha256(file_path.encode()).hexdigest()

        hash_sha256 = hashlib.sha256()

        if os.path.isfile(file_path):
            # Handle single file
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        elif os.path.isdir(file_path):
            # Handle directory (like TensorFlow saved models)
            for root, dirs, files in os.walk(file_path):
                # Sort for consistent hashing
                dirs.sort()
                files.sort()

                for file in files:
                    file_full_path = os.path.join(root, file)
                    # Add relative path to hash for consistency
                    relative_path = os.path.relpath(file_full_path, file_path)
                    hash_sha256.update(relative_path.encode())

                    try:
                        with open(file_full_path, "rb") as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hash_sha256.update(chunk)
                    except (PermissionError, OSError) as e:
                        # If we can't read a file, include its name and size in hash
                        logger.warning(f"Cannot read file {file_full_path}: {e}")
                        hash_sha256.update(f"UNREADABLE:{relative_path}:{os.path.getsize(file_full_path)}".encode())
        else:
            # Fallback for other types
            hash_sha256.update(f"UNKNOWN_TYPE:{file_path}".encode())

        return hash_sha256.hexdigest()

    def _check_data_reuse(self, experiment_id: str, pipeline_hash: str,
                          force_reload: bool) -> Tuple[bool, Optional[str]]:
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

    # Repository methods (write operations)
    def create_experiment(self, name: str, config: Dict[str, Any], description: str = "") -> str:
        return self.repository.create_experiment(name, config, description)

    def create_run(self, experiment_id: str, config: Dict[str, Any], force_reload: bool = False) -> str:
        return self.repository.create_run(experiment_id, config, force_reload)

    def complete_run(self, run_id: str, status: str = "completed") -> None:
        return self.repository.complete_run(run_id, status)

    def store_artifact(self, run_id: str, file_path: str, artifact_type: str,
                       artifact_role: str, stage: str, metadata: Dict[str, Any] = None) -> str:
        return self.repository.store_artifact(run_id, file_path, artifact_type, artifact_role, stage, metadata)

    def store_metrics(self, run_id: str, metrics: Dict[str, Any], stage: str,
                      epoch: int = None, metric_type: str = "predefined",
                      metric_category: str = "custom") -> None:
        return self.repository.store_metrics(run_id, metrics, stage, epoch, metric_type, metric_category)

    def store_parameters(self, run_id: str, parameters: Dict[str, Any], stage: str) -> None:
        return self.repository.store_parameters(run_id, parameters, stage)

    def record_data_transformation(self, run_id: str, input_hash: str, output_hash: str,
                                   transformation_type: str, transformation_config: Dict[str, Any],
                                   stage: str) -> None:
        return self.repository.record_data_transformation(run_id, input_hash, output_hash, transformation_type,
                                                          transformation_config, stage)

    # Query methods (read operations)
    def find_experiment_by_name_or_id(self, identifier: str) -> Optional[Dict[str, Any]]:
        return self.query_service.find_experiment_by_name_or_id(identifier)

    def get_experiments(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.query_service.get_experiments(limit)

    def get_runs_for_experiment(self, experiment_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        return self.query_service.get_runs_for_experiment(experiment_id, limit)

    def get_run_lineage(self, run_id: str) -> Dict[str, Any]:
        return self.query_service.get_run_lineage(run_id)

    def get_reusable_artifacts(self, experiment_id: str, pipeline_hash: str) -> Dict[str, str]:
        return self.query_service.get_reusable_artifacts(experiment_id, pipeline_hash)

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        return self.query_service.compare_runs(run_ids)

    def query_artifacts_by_criteria(self, artifact_type: str = None, stage: str = None,
                                    limit: int = 20) -> List[Dict[str, Any]]:
        return self.query_service.query_artifacts_by_criteria(artifact_type, stage, limit)

    def get_database_summary(self) -> Dict[str, Any]:
        return self.query_service.get_database_summary()

    def delete_run_cascade(self, run_id: str) -> bool:
        return self.query_service.delete_run_cascade(run_id)

    def get_training_progress(self, run_id: str) -> Dict[str, Any]:
        return self.query_service.get_training_progress(run_id)

    def __del__(self):
        """Clean up database connection on object destruction."""
        if self.connection:
            self.connection.close()
