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
from typing import Dict, Any, Optional, Tuple, List

from src.tof_ml.data.data_manager import DataManager
from src.tof_ml.data.data_provenance import ProvenanceTracker
from src.tof_ml.models.model_factory import ModelFactory
from src.tof_ml.training.trainer import Trainer
from src.tof_ml.reporting.report_generator import ReportGenerator
from src.tof_ml.database.api import DBApi

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
        self.output_dir = self._setup_output_directory()
        
        # Initialize components
        self.data_manager = None
        self.model_factory = None
        self.trainer = None
        self.report_generator = None
        self.provenance_tracker = ProvenanceTracker(self.config)
        
        # Initialize database connection if enabled
        self.db_api = None
        if self.config.get("use_database", True):
            db_config_path = self.config.get("database", {}).get("config_path", "config/database_config.yaml")
            self.db_api = DBApi(config_path=db_config_path)
        
        # Initialize metadata
        self.metadata = {
            "pipeline_start_time": datetime.datetime.now().isoformat(),
            "config": self.config,
        }
        
        logger.info(f"Pipeline initialized with output directory: {self.output_dir}")
    
    def _setup_output_directory(self) -> str:
        """Set up the output directory structure."""
        base_output_dir = self.config.get("output_dir", "./output")
        
        # Use experiment name if provided, otherwise use timestamp
        experiment_name = self.config.get("experiment_name")
        timestamp_str = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        
        if experiment_name:
            unique_subdir = f"{experiment_name}_{timestamp_str}"
        else:
            unique_subdir = f"run_{timestamp_str}"
        
        full_output_path = os.path.join(base_output_dir, unique_subdir)
        os.makedirs(full_output_path, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(full_output_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(full_output_path, "data"), exist_ok=True)
        os.makedirs(os.path.join(full_output_path, "reports"), exist_ok=True)
        os.makedirs(os.path.join(full_output_path, "logs"), exist_ok=True)
        
        return full_output_path
    
    def _load_class_dynamically(self, class_mapping: Dict[str, str], key: str) -> type:
        """
        Dynamically load a class based on the class mapping configuration.
        
        Args:
            class_mapping: Dictionary mapping class keys to fully qualified class names
            key: The key to use in the class mapping
            
        Returns:
            The loaded class
        """
        class_str = class_mapping.get(key)
        if not class_str:
            raise ValueError(f"Class mapping for key '{key}' not found")
        
        module_name, class_name = class_str.rsplit('.', 1)
        module = importlib.import_module(module_name)
        loaded_class = getattr(module, class_name)
        
        return loaded_class
    
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
        
        logger.info("Pipeline run completed successfully")
        return self.metadata
    
    def run_training(self):
        """Run only the training part of the pipeline."""
        logger.info("Starting training-only pipeline run")
        
        # Load and process data
        self._load_and_process_data()
        
        # Create and train model
        self._create_and_train_model()
        
        logger.info("Training completed successfully")
        return self.metadata
    
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
        
        import json
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
            
            # Load and process data
            self.data_manager.load_data()
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
            # Initialize model factory
            self.model_factory = ModelFactory(self.config)
            
            # Create model
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
            self.metadata.update({
                "model_info": self.model_factory.get_metadata(),
                "training_info": self.trainer.get_metadata(),
                "training_results": training_results
            })
        
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
            
            # Save evaluation results
            import json
            results_path = os.path.join(self.output_dir, "evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Model evaluation completed with test_loss: {evaluation_results.get('test_loss', 'N/A')}")
        return evaluation_results
    
    def _generate_report(self, results):
        """Generate reports from the evaluation results."""
        logger.info("Generating reports")
        
        # Track report generation
        with self.provenance_tracker.track_operation("report_generation"):
            # Initialize report generator
            self.report_generator = ReportGenerator(
                config=self.config,
                data_manager=self.data_manager,
                training_results=self.metadata.get("training_results", {}),
                evaluation_results=results,
                output_dir=os.path.join(self.output_dir, "reports")
            )
            
            # Generate reports
            report_paths = self.report_generator.generate_reports()
            
            # Update metadata
            self.metadata.update({
                "report_paths": report_paths
            })
        
        logger.info(f"Report generation completed, {len(report_paths)} reports generated")
    
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
