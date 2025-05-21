#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline orchestrator for the ML Provenance Tracker Framework.
This module coordinates the execution of the ML pipeline with provenance tracking.
"""

import os
import logging
import datetime
import importlib
import yaml
import json
from typing import Dict, Any, Optional, Tuple, List

from src.tof_ml.data.data_manager import DataManager
from src.tof_ml.data.data_provenance import ProvenanceTracker
from src.tof_ml.models.model_factory import ModelFactory
from src.tof_ml.training.trainer import Trainer
from src.tof_ml.reporting.report_generator import BaseReportGenerator
from src.tof_ml.database.api import DBApi
from src.tof_ml.pipeline.plugins.registry import PluginRegistry, get_registry

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Orchestrates the machine learning pipeline with provenance tracking.
    
    This class is responsible for coordinating all the components of the pipeline:
    - Data loading and transformation
    - Model creation and training
    - Evaluation and reporting
    - Database integration and provenance tracking
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Complete configuration dictionary for the pipeline
        """
        self.config = config

        # Initialize metadata
        self.metadata = {
            "pipeline_start_time": datetime.datetime.now().isoformat(),
            "config": self.config,
        }

        # Set up directory structure
        self.output_dir = self._setup_output_directory()
        self.experiment_dir = os.path.dirname(os.path.dirname(self.output_dir))

        # Initialize provenance tracker
        self.provenance_tracker = ProvenanceTracker(self.config)

        # Synchronize run_id with provenance tracker
        if hasattr(self.provenance_tracker, 'run_id'):
            self.metadata["run_id"] = self.provenance_tracker.run_id

        # Initialize components
        self.data_manager = None
        self.model_factory = None
        self.trainer = None
        self.report_generator = None

        # Initialize plugin registry and load plugins from config
        self.plugin_registry = get_registry()
        self._load_class_mapping()

        # Initialize database connection if enabled
        self.db_api = None
        if self.config.get("use_database", True):
            db_config_path = self.config.get("database", {}).get("config_path", "config/database_config.yaml")
            self.db_api = DBApi(config_path=db_config_path)

        logger.info(f"Pipeline initialized with output directory: {self.output_dir}")

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

        # Store run_id in metadata for easier reference
        self.metadata = self.metadata or {}
        self.metadata["run_id"] = run_id
        self.metadata["experiment_dir"] = experiment_dir

        return run_dir
    
    def _load_class_mapping(self) -> None:
        """Load class mapping from configuration and register plugins."""
        mapping_path = self.config.get("class_mapping_path", "config/class_mapping_config.yaml")

        with open(mapping_path, 'r') as f:
            class_mapping = yaml.safe_load(f)
        
        # Register plugins from class mapping
        self.plugin_registry.register_plugins_from_config(class_mapping)

    def run_pipeline(self):
        """Run the complete pipeline."""
        logger.info("Starting complete pipeline run")
        
        # Track provenance for the entire pipeline
        with self.provenance_tracker.track_operation("pipeline_run"):
            # Load and process data
            self._load_and_process_data()
            
            # Create and train model
            self._create_and_train_model()
            
            # Evaluate model
            results = self._evaluate_model()
            
            # Generate report
            self._generate_report(results)
            
            # Save to database if enabled
            if self.db_api:
                self._save_to_database(results)

            self._save_provenance_information()
        
        logger.info("Pipeline run completed successfully")
        return self.metadata
    
    def run_training(self):
        """Run only the training part of the pipeline..
        This should really only be used if we have some way
        To identify the last saved data, splits, etc and the full
        provenance and then it makes sense to just run training.."""
        pass
    
    def run_evaluation(self):
        """Run only the evaluation part of the pipeline."""
        logger.info("Starting evaluation-only pipeline run")
        
        # Load and process data
        self._load_and_process_data()
        
        # Load pre-trained model
        model_path = self.config.get("evaluation", {}).get("model_path")
        if not model_path:
            raise ValueError("Model path must be provided for evaluation-only mode")
        
        # Initialize model factory
        self.model_factory = ModelFactory(self.config)
        
        # Load pretrained model
        model = self.model_factory.load_model(model_path)
        
        # Evaluate model
        results = self._evaluate_model(model=model)
        
        # Generate report
        self._generate_report(results)
        
        logger.info("Evaluation completed successfully")
        return self.metadata
    
    def generate_report(self):
        """Generate reports from existing data and results."""
        logger.info("Starting report-only pipeline run")
        
        # Load evaluation results
        results_path = self.config.get("reporting", {}).get("results_path")
        if not results_path:
            raise ValueError("Results path must be provided for report-only mode")

        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Generate report
        self._generate_report(results)
        
        logger.info("Report generation completed successfully")
        return self.metadata
    
    def _load_and_process_data(self):
        """Load and process data using the data manager."""
        logger.info("Loading and processing data")
        
        # Track data loading and processing
        with self.provenance_tracker.track_operation("data_processing"):
            # Initialize data manager
            self.data_manager = DataManager(
                config=self.config,
                output_dir=os.path.join(self.output_dir, "data"),
                provenance_tracker=self.provenance_tracker
            )
            
            # Get the data loader plugin name from config
            data_loader_name = self.config.get("plugins", {}).get("data_loader")
            
            if data_loader_name:
                logger.info(f"Using data loader plugin: {data_loader_name}")
                # Get data loader specific config
                loader_config = self.config.get("data_loader", {})
                
                try:
                    # Get loader from plugin registry
                    loader = self.plugin_registry.get_data_loader(
                        data_loader_name, 
                        config=self.config,
                        **loader_config
                    )
                    
                    # Set the loader in the data manager
                    self.data_manager.set_loader(loader)
                except KeyError:
                    logger.warning(f"Data loader plugin {data_loader_name} not found, using default")
            
            # Load and process data
            self.data_manager.load_data()
            # Check if we're reusing data (datasets will be loaded but raw_data may be None)
            data_reused = self.data_manager.metadata.get("data_reused", False)

            # Only preprocess and split if we're not reusing data
            if not data_reused:
                # Continue with preprocessing and splitting
                self.data_manager.preprocess_data()
                self.data_manager.split_data()

                # Save processed datasets
                self.data_manager.save_datasets()
            
            # Update metadata
            self.metadata.update({
                "data_info": self.data_manager.get_metadata(),
                "data_provenance": self.provenance_tracker.get_data_provenance()
            })
        
        logger.info("Data loading and processing completed")
    
    def _create_and_train_model(self):
        """Create and train a machine learning model."""
        logger.info("Creating and training model")
        
        # Track model training
        with self.provenance_tracker.track_operation("model_training"):
            # Get the model plugin name from config
            model_name = self.config.get("plugins", {}).get("model")
            model = None
            
            if model_name:
                logger.info(f"Using model plugin: {model_name}")
                # Get model specific config
                model_config = self.config.get("model", {})
                
                try:
                    # Remove non-model parameters
                    model_params = model_config.copy()
                    for key in ["type", "epochs", "batch_size", "early_stopping", "checkpoint"]:
                        model_params.pop(key, None)
                    
                    # Get model from plugin registry
                    model = self.plugin_registry.get_model(
                        model_name,
                        **model_params
                    )
                except KeyError:
                    logger.warning(f"Model plugin {model_name} not found, using ModelFactory")
            
            if model is None:
                # Fallback to model factory
                self.model_factory = ModelFactory(self.config)
                model = self.model_factory.create_model()
            
            # Initialize trainer
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
            model_info = {}
            if model_name:
                model_info = {k: v for k, v in self.config.get("model", {}).items() if k != "type"}
                model_info["plugin"] = model_name
            else:
                model_info = self.model_factory.get_metadata()
                
            self.metadata.update({
                "model_info": model_info,
                "training_info": self.trainer.get_metadata(),
                "training_results": training_results
            })
            artifacts = {
                "model_path": self.metadata.get("training_info", {}).get("best_model_path", ""),
                "final_model_path": self.metadata.get("training_info", {}).get("final_model_path", "")
            }
            self.provenance_tracker.update_stage_completion("model_training", True, artifacts)
        
        logger.info(f"Model training completed with final val_loss: {training_results.get('val_loss', 'N/A')}")
    
    def _evaluate_model(self, model=None):
        """Evaluate the trained model."""
        logger.info("Evaluating model")
        
        # Track model evaluation
        with self.provenance_tracker.track_operation("model_evaluation"):
            if model is None and self.trainer is None:
                raise ValueError("Either model or trainer must be provided for evaluation")
            
            if model is None:
                model = self.trainer.model
            
            # Evaluate model
            evaluation_results = self.trainer.evaluate(model)
            
            # Update metadata
            self.metadata.update({
                "evaluation_results": evaluation_results
            })

            results_path = os.path.join(self.output_dir, "reports", "evaluation_results.json")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            artifacts = {
                "evaluation_results": results_path
            }
            self.provenance_tracker.update_stage_completion("model_evaluation", True, artifacts)
        
        logger.info(f"Model evaluation completed with test_loss: {evaluation_results.get('test_loss', 'N/A')}")
        return evaluation_results
    
    def _generate_report(self, results):
        """Generate reports from the evaluation results."""
        logger.info("Generating reports")
        
        # Track report generation
        with self.provenance_tracker.track_operation("report_generation"):
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
                        output_dir=os.path.join(self.output_dir, "reports")
                    )
                except KeyError:
                    logger.warning(f"Report generator plugin {report_generator_name} not found, using default")
            
            if report_generator is None:
                # Use default report generator
                report_generator = BaseReportGenerator(
                    config=self.config,
                    data_manager=self.data_manager,
                    training_results=self.metadata.get("training_results", {}),
                    evaluation_results=results,
                    output_dir=os.path.join(self.output_dir, "reports")
                )
            
            # Generate reports
            report_paths = report_generator.generate_reports()
            
            # Update metadata
            self.metadata.update({
                "report_paths": report_paths
            })
        
        logger.info(f"Report generation completed, {len(report_paths)} reports generated")

    def _save_provenance_information(self):
        """Save and visualize provenance information."""
        logger.info("Saving provenance information")

        # Create provenance directory
        provenance_dir = os.path.join(self.experiment_dir, "provenance")
        os.makedirs(provenance_dir, exist_ok=True)

        # Save provenance graph to JSON
        json_path = os.path.join(provenance_dir, f"lineage_graph_{self.metadata['run_id']}.json")
        provenance_path = self.provenance_tracker.save_provenance_graph(json_path)

        # Add to metadata
        self.metadata["provenance_graph"] = provenance_path

        # Visualize provenance graph (if graphviz is available)
        try:
            svg_path = os.path.join(provenance_dir, f"lineage_graph_{self.metadata['run_id']}.svg")
            visualization_path = self.provenance_tracker.visualize_provenance_graph(svg_path)

            if visualization_path:
                self.metadata["provenance_visualization"] = visualization_path
                logger.info(f"Provenance graph visualization saved to {visualization_path}")
        except Exception as e:
            logger.warning(f"Could not visualize provenance graph: {e}")
            logger.warning("To enable visualization, install graphviz and ensure it's in your PATH")

        logger.info(f"Provenance graph saved to {provenance_path}")
    
    def _save_to_database(self, results):
        """Save experiment results to the database."""
        logger.info("Saving results to database")
        
        if not self.db_api:
            logger.warning("Database API not initialized, skipping database save")
            return
        
        # Create metadata dictionary with important metrics
        db_metadata = {
            "experiment_id": self.metadata.get("pipeline_start_time"),
            "experiment_name": self.config.get("experiment_name", "unnamed_experiment"),
            "model_type": self.config.get("model", {}).get("type", "unknown"),
            "dataset_name": self.config.get("data", {}).get("dataset_name", "unknown"),
            "dataset_size": self.data_manager.get_metadata().get("total_samples", 0),
            "training_time": self.metadata.get("training_info", {}).get("training_time", 0),
            "best_val_loss": results.get("val_loss", float('inf')),
            "test_loss": results.get("test_loss", float('inf')),
            "test_metrics": {k: v for k, v in results.items() if k != "test_loss" and k != "val_loss"}
        }
        
        # Save to database
        experiment_id = self.db_api.record_experiment(
            metadata=db_metadata,
            config=self.config,
            training_results=self.metadata.get("training_results", {}),
            evaluation_results=results,
            model_path=self.metadata.get("training_info", {}).get("best_model_path", ""),
            report_paths=self.metadata.get("report_paths", {})
        )
        
        logger.info(f"Results saved to database with experiment ID: {experiment_id}")
        
        # Update metadata
        self.metadata["database_experiment_id"] = experiment_id