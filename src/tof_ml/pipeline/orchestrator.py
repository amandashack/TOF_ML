#!/usr#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Pipeline orchestrator with comprehensive artifact tracking and database integration.
"""

import os
import logging
import datetime
import yaml
import json
from typing import Dict, Any, List

from src.tof_ml.data.data_manager import DataManager
from src.tof_ml.data.data_provenance import ProvenanceTracker
from src.tof_ml.training.trainer import Trainer
from src.tof_ml.database.api import DBApi
from src.tof_ml.reporting.report_generator import ReportGeneratorPlugin
from src.tof_ml.pipeline.registry import get_registry

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Enhanced pipeline orchestrator with comprehensive artifact tracking.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced pipeline orchestrator."""
        self.config = config

        # Initialize metadata
        self.metadata = {
            "pipeline_start_time": datetime.datetime.now().isoformat(),
            "config": self.config,
        }

        # Set up directory structure
        self.output_dir = self._setup_output_directory()
        self.experiment_dir = os.path.dirname(os.path.dirname(self.output_dir))

        # Initialize database API
        self.db_api = None
        if self.config.get("use_database", True):
            db_config_path = self.config.get("database", {}).get("config_path", "config/database_config.yaml")
            self.db_api = DBApi(config_path=db_config_path)

        # Initialize provenance tracker with database integration
        self.provenance_tracker = ProvenanceTracker(self.config, self.db_api)

        # Synchronize metadata with provenance tracker
        self.metadata.update({
            "experiment_id": self.provenance_tracker.get_experiment_id(),
            "run_id": self.provenance_tracker.get_run_id()
        })

        # Initialize components
        self.data_manager = None
        self.trainer = None
        self.report_generator = None

        # Initialize plugin registry and load plugins from config
        self.plugin_registry = get_registry()
        self._load_class_mapping()

        logger.info(f"Enhanced Pipeline initialized with output directory: {self.output_dir}")

    def _setup_output_directory(self) -> str:
        """Set up the output directory structure."""
        base_output_dir = self.config.get("output_dir", "./output")

        # Get experiment name, use default if not provided
        experiment_name = self.config.get("experiment_name", "unnamed_experiment")

        # Get run ID with timestamp
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp_str}"

        # Create experiment directory
        experiment_dir = os.path.join(base_output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        # Create runs directory
        runs_dir = os.path.join(experiment_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)

        # Create run-specific directory
        run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Create run subdirectories
        os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        return run_dir

    def _load_class_mapping(self) -> None:
        """Load class mapping from configuration and register plugins."""
        mapping_path = self.config.get("class_mapping_path", "config/class_mapping_config.yaml")

        with open(mapping_path, 'r') as f:
            class_mapping = yaml.safe_load(f)

        # Register plugins from class mapping
        self.plugin_registry.register_plugins_from_config(class_mapping)

    def run_pipeline(self):
        """Run the complete pipeline with comprehensive tracking."""
        logger.info("Starting complete pipeline run")

        try:
            # Load and process data
            self._load_and_process_data()

            # Create and train model
            self._create_and_train_model()

            # Evaluate model
            results = self._evaluate_model()

            # Generate report
            self._generate_report(results)

            # Complete the run successfully
            self.provenance_tracker.complete_run("completed")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.provenance_tracker.complete_run("failed")
            raise

        logger.info("Pipeline run completed successfully")
        return self.metadata

    def run_training(self):
        """Run only the training part of the pipeline."""
        logger.info("Starting training-only pipeline run")

        try:
            # Load existing data
            self._load_and_process_data()

            # Create and train model
            self._create_and_train_model()

            # Complete the run
            self.provenance_tracker.complete_run("completed")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.provenance_tracker.complete_run("failed")
            raise

        logger.info("Training completed successfully")
        return self.metadata

    def run_evaluation(self):
        """Run only the evaluation part of the pipeline."""
        logger.info("Starting evaluation-only pipeline run")

        try:
            # Load and process data
            self._load_and_process_data()

            # Load pre-trained model
            model_path = self.config.get("evaluation", {}).get("model_path")
            if not model_path:
                raise ValueError("Model path must be provided for evaluation-only mode")

            # Initialize model factory and load model
            # This would need to be implemented based on your model loading strategy

            # Evaluate model
            results = self._evaluate_model()

            # Generate report
            self._generate_report(results)

            # Complete the run
            self.provenance_tracker.complete_run("completed")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.provenance_tracker.complete_run("failed")
            raise

        logger.info("Evaluation completed successfully")
        return self.metadata

    def generate_report(self):
        """Generate reports from existing data and results."""
        logger.info("Starting report-only pipeline run")

        try:
            # Load evaluation results
            results_path = self.config.get("reporting", {}).get("results_path")
            if not results_path:
                raise ValueError("Results path must be provided for report-only mode")

            with open(results_path, 'r') as f:
                results = json.load(f)

            # Generate report
            self._generate_report(results)

            # Complete the run
            self.provenance_tracker.complete_run("completed")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            self.provenance_tracker.complete_run("failed")
            raise

        logger.info("Report generation completed successfully")
        return self.metadata

    def _load_and_process_data(self):
        """Load and process data with artifact tracking - FIXED VERSION."""
        logger.info("Loading and processing data")

        self.data_manager = DataManager(
            config=self.config,
            output_dir=os.path.join(self.output_dir, "data"),
            provenance_tracker=self.provenance_tracker
        )

        # Get the data loader plugin name from config
        data_loader_name = self.config.get("plugins", {}).get("data_loader")

        if data_loader_name:
            logger.info(f"Using data loader plugin: {data_loader_name}")
            # Get data loader specific config - this should include transformations
            loader_config = self.config.get("transformations", {})  # Use transformations config

            try:
                # Get loader from plugin registry
                loader = self.plugin_registry.get_data_loader(
                    data_loader_name,
                    config=self.config,
                    **loader_config  # Pass transformation config as kwargs
                )

                # Set the loader in the data manager
                self.data_manager.set_loader(loader)

                logger.info(f"Successfully loaded plugin: {data_loader_name}")

            except KeyError as e:
                logger.error(f"Data loader plugin {data_loader_name} not found: {e}")
        else:
            logger.warning("No data loader specified in plugins")

        # Load data (will check for reuse automatically)
        self.data_manager.load_data()

        # Check if we're reusing data
        data_reused = self.data_manager.data_reused

        # Only preprocess and split if we're not reusing data
        if not data_reused:
            # Continue with preprocessing and splitting
            self.data_manager.preprocess_data()
            self.data_manager.split_data()

        # Update metadata
        self.metadata.update({
            "data_info": self.data_manager.get_data_summary()
        })

        logger.info("Data loading and processing completed")

    def _create_and_train_model(self):
        """Create and train a machine learning model with artifact tracking."""
        logger.info("Creating and training model")

        # Get the model plugin name from config
        model_name = self.config.get("plugins", {}).get("model")

        if not model_name:
            raise ValueError("No model specified in plugins configuration")

        logger.info(f"Using model plugin: {model_name}")

        # Get model specific config
        model_config = self.config.get("model", {})

        # Remove non-model parameters
        model_params = {k: v for k, v in model_config.items()
                        if k not in ["type", "epochs", "batch_size", "early_stopping", "checkpoint"]}

        try:
            # Get model from plugin registry
            model = self.plugin_registry.get_model(model_name, **model_params)
        except KeyError:
            raise ValueError(f"Model plugin '{model_name}' not found in registry. "
                             f"Available models: {self.plugin_registry.list_models()}")

        # Initialize enhanced trainer
        self.trainer = Trainer(
            config=self.config,
            model=model,
            data_manager=self.data_manager,
            output_dir=os.path.join(self.output_dir, "models"),
            provenance_tracker=self.provenance_tracker
        )

        # Train model
        training_results = self.trainer.train()

        # Update metadata
        self.metadata.update({
            "model_info": {
                "plugin": model_name,
                "metadata": model.get_metadata() if hasattr(model, 'get_metadata') else {}
            },
            "training_info": self.trainer.get_training_summary(),
            "training_results": training_results
        })

        logger.info(f"Model training completed with final val_loss: {training_results.get('best_val_loss', 'N/A')}")

    def _evaluate_model(self, model=None):
        """Evaluate the trained model with artifact tracking."""
        logger.info("Evaluating model")

        if model is None and self.trainer is None:
            raise ValueError("Either model or trainer must be provided for evaluation")

        if model is None:
            model = self.trainer.model

        # Evaluate model using trainer
        evaluation_results = self.trainer.evaluate(model)

        # Update metadata
        self.metadata.update({
            "evaluation_results": evaluation_results
        })

        logger.info(f"Model evaluation completed with test_loss: {evaluation_results.get('test_loss', 'N/A')}")
        return evaluation_results

    def _generate_report(self, results):
        """Generate reports with artifact tracking."""
        logger.info("Generating reports")

        reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        with self.provenance_tracker.track_stage("report_generation", self.config.get("reporting", {})):
            # Get the report generator plugin name from config
            report_generator_name = self.config.get("plugins", {}).get("report_generator")
            report_generator = None

            if report_generator_name:
                logger.info(f"Using report generator plugin: {report_generator_name}")

                try:
                    # Get report generator from plugin registry
                    report_generator = self.plugin_registry.get_report_generator(
                        report_generator_name,
                        config=self.config,
                        data_manager=self.data_manager,
                        training_results=self.metadata.get("training_results", {}),
                        evaluation_results=results,
                        output_dir=reports_dir
                    )
                except KeyError:
                    logger.warning(f"Report generator plugin {report_generator_name} not found, using default")

            if report_generator is None:
                # Use default report generator
                report_generator = ReportGeneratorPlugin(
                    config=self.config,
                    data_manager=self.data_manager,
                    training_results=self.metadata.get("training_results", {}),
                    evaluation_results=results,
                    output_dir=reports_dir
                )

            # Generate reports
            report_paths = report_generator.generate_reports()

            # Record report artifacts
            for report_type, report_path in report_paths.items():
                if report_path and os.path.exists(report_path):
                    self.provenance_tracker.record_artifact(
                        report_path,
                        "report",
                        report_type,
                        {
                            "report_generator": report_generator_name or "default",
                            "file_size": os.path.getsize(report_path)
                        }
                    )

            # Update metadata
            self.metadata.update({
                "report_paths": report_paths
            })

        logger.info(f"Report generation completed, {len(report_paths)} reports generated")

    def get_run_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current run."""
        if not self.provenance_tracker:
            return self.metadata

        # Get lineage information
        lineage = self.provenance_tracker.get_run_lineage()

        # Combine with metadata
        summary = {
            "metadata": self.metadata,
            "lineage": lineage,
            "experiment_id": self.provenance_tracker.get_experiment_id(),
            "run_id": self.provenance_tracker.get_run_id()
        }

        return summary

    def compare_with_previous_runs(self, run_ids: List[str] = None) -> Dict[str, Any]:
        """Compare current run with previous runs."""
        if not self.db_api:
            logger.warning("Database API not available for comparison")
            return {}

        current_run_id = self.provenance_tracker.get_run_id()

        if run_ids is None:
            # Get recent runs from same experiment
            experiment_id = self.provenance_tracker.get_experiment_id()
            cursor = self.db_api.connection.cursor()
            cursor.execute(
                "SELECT id FROM runs WHERE experiment_id = ? AND id != ? ORDER BY created_at DESC LIMIT 5",
                (experiment_id, current_run_id)
            )
            run_ids = [row["id"] for row in cursor.fetchall()]

        if not run_ids:
            logger.info("No previous runs found for comparison")
            return {}

        # Include current run in comparison
        all_run_ids = [current_run_id] + run_ids

        return self.db_api.compare_runs(all_run_ids)

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all runs in the current experiment."""
        if not self.db_api:
            return {}

        experiment_id = self.provenance_tracker.get_experiment_id()
        cursor = self.db_api.connection.cursor()

        # Get all runs for this experiment
        cursor.execute(
            "SELECT id, created_at, status, data_reused FROM runs WHERE experiment_id = ? ORDER BY created_at DESC",
            (experiment_id,)
        )
        runs = [dict(row) for row in cursor.fetchall()]

        # Get experiment info
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        experiment_info = dict(cursor.fetchone() or {})

        return {
            "experiment": experiment_info,
            "runs": runs,
            "total_runs": len(runs),
            "successful_runs": len([r for r in runs if r["status"] == "completed"]),
            "data_reused_runs": len([r for r in runs if r["data_reused"]])
        }