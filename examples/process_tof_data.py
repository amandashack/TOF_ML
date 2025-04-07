#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for processing ToF data using the Enhanced H5 Loader.

This script demonstrates how to:
1. Load raw ToF data from multiple H5 files
2. Apply preprocessing transformations
3. Split data into train/validation/test sets
4. Save preprocessed data
5. Train a model (optional)
6. Track provenance throughout the process

Usage:
    python examples/process_tof_data.py --config config/tof_data_config.yaml
"""

import os
import sys
import argparse
import logging
import yaml
import datetime
from typing import Dict, Any

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tof_ml.data.data_manager import DataManager
from src.tof_ml.data.data_provenance import ProvenanceTracker
from src.tof_ml.models.model_factory import ModelFactory
from src.tof_ml.training.trainer import Trainer
from src.tof_ml.reporting.report_generator import ReportGenerator
from src.tof_ml.database.api import DBApi
from src.tof_ml.data.column_mapping import COLUMN_MAPPING, REDUCED_COLUMN_MAPPING
from src.tof_ml.visualization.pipeline_visualizer import visualize_provenance_graph

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process ToF data")
    
    # Configuration options
    parser.add_argument("--config", type=str, default="config/tof_data_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data-path", type=str,
                        help="Path to ToF data directory (overrides config)")
    parser.add_argument("--output-dir", type=str,
                        help="Output directory (overrides config)")
    
    # Processing options
    parser.add_argument("--mid1", type=str,
                        help="Mid1 voltage configuration (comma-separated, e.g., '0.2,0.2')")
    parser.add_argument("--mid2", type=str,
                        help="Mid2 voltage configuration (comma-separated, e.g., '0.2,0.2')")
    parser.add_argument("--n-samples", type=int,
                        help="Number of samples to use (overrides config)")
    parser.add_argument("--batch-size", type=int,
                        help="Batch size for loading data (overrides config)")
    
    # Pipeline control
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only preprocess data, don't train model")
    parser.add_argument("--train-model", action="store_true",
                        help="Train model after preprocessing")
    
    # Database options
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip database logging")
    
    return parser.parse_args()

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("tof_data_processing.log")
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_config_from_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Update configuration with command-line arguments."""
    # Data options
    if args.data_path:
        config["data"]["dataset_path"] = args.data_path
    
    if args.n_samples:
        config["data"]["n_samples"] = args.n_samples
    
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    
    # Voltage configurations
    if args.mid1:
        config["data"]["mid1"] = [float(v) for v in args.mid1.split(",")]
    
    if args.mid2:
        config["data"]["mid2"] = [float(v) for v in args.mid2.split(",")]
    
    # Output options
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Database options
    if args.skip_db:
        config["use_database"] = False
    
    return config

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command-line arguments
    # config = update_config_from_args(config, args)
    
    # Set up output directory
    output_dir = config.get("output_dir", "./output")
    experiment_name = config.get("experiment_name", "tof_data_processing")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Update config with experiment directory
    config["output_dir"] = experiment_dir
    
    # Initialize provenance tracker
    provenance_tracker = ProvenanceTracker(config)
    
    # Initialize data manager
    data_config = config.get("data", {})
    mid1 = data_config.get("mid1", [0.0, 1])
    mid2 = data_config.get("mid2", [0.0, 1])
    
    data_manager = DataManager(
        config=config,
        output_dir=os.path.join(experiment_dir, "data"),
        provenance_tracker=provenance_tracker
    )
    
    # Load data with column mapping
    logging.info("Loading ToF data...")
    data_loader_params = {
        "mid1": mid1,
        "mid2": mid2,
        "column_mapping": COLUMN_MAPPING,
        "reduced_column_mapping": REDUCED_COLUMN_MAPPING
    }
    
    # Inject loader parameters into config
    for key, value in data_loader_params.items():
        data_config[key] = value
    
    # Load data
    data_manager.load_data()
    
    # Preprocess data
    logging.info("Preprocessing data...")
    data_manager.preprocess_data()
    
    # Split data
    logging.info("Splitting data...")
    data_manager.split_data()
    
    # Save preprocessed datasets
    logging.info("Saving preprocessed datasets...")
    saved_paths = data_manager.save_datasets()
    
    # Save provenance graph
    provenance_path = provenance_tracker.save_provenance_graph(
        os.path.join(experiment_dir, "provenance", "lineage_graph.json")
    )
    
    # Create visualizations directory
    vis_dir = os.path.join(experiment_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate custom visualization of the pipeline
    svg_path = visualize_provenance_graph(
        provenance_path,
        os.path.join(vis_dir, "pipeline_visualization.svg")
    )
    logging.info(f"Custom pipeline visualization saved to: {svg_path}")
    
    # Train model if requested
    training_results = None
    evaluation_results = None
    
    if args.train_model and not args.preprocess_only:
        logging.info("Training model...")
        
        # Initialize model factory
        model_factory = ModelFactory(config)
        
        # Create model
        model = model_factory.create_model()
        
        # Initialize trainer
        trainer = Trainer(
            config=config,
            model=model,
            data_manager=data_manager,
            output_dir=os.path.join(experiment_dir, "models"),
            provenance_tracker=provenance_tracker
        )
        
        # Train model
        training_results = trainer.train()
        
        # Evaluate model
        evaluation_results = trainer.evaluate()
        
        # Generate reports
        report_generator = ReportGenerator(
            config=config,
            data_manager=data_manager,
            training_results=training_results,
            evaluation_results=evaluation_results,
            output_dir=os.path.join(experiment_dir, "reports")
        )
        
        report_paths = report_generator.generate_reports()
        saved_paths.update(report_paths)
    
    # Save to database
    if config.get("use_database", True):
        logging.info("Saving to database...")
        db_config_path = config.get("database", {}).get("config_path", "config/database_config.yaml")
        db_api = DBApi(config_path=db_config_path)
        
        # Create metadata for database
        metadata = {
            "experiment_id": timestamp,
            "experiment_name": experiment_name,
            "dataset_name": os.path.basename(config["data"]["dataset_path"]),
            "preprocessing_steps": [t["type"] for t in config["preprocessing"]["transformers"]],
            "total_samples": data_manager.get_metadata().get("total_samples", 0),
            "mid1": mid1,
            "mid2": mid2
        }
        
        if training_results:
            metadata.update({
                "training_time": training_results.get("training_time", 0),
                "epochs_completed": training_results.get("epochs_completed", 0),
                "best_val_loss": training_results.get("best_val_loss", float('inf'))
            })
        
        # Record experiment
        experiment_id = db_api.record_experiment(
            metadata=metadata,
            config=config,
            training_results=training_results,
            evaluation_results=evaluation_results,
            model_path=training_results["best_model_path"] if training_results else None,
            report_paths=saved_paths
        )
        
        logging.info(f"Experiment recorded in database with ID: {experiment_id}")
    
    logging.info(f"Processing completed. Results saved to {experiment_dir}")
    logging.info(f"Saved datasets: {list(saved_paths.keys())}")
    logging.info(f"Provenance graph: {provenance_path}")
    logging.info(f"Pipeline visualization: {svg_path}")
    
    if training_results:
        logging.info(f"Best model: {training_results.get('best_model_path', 'N/A')}")
        logging.info(f"Test metrics: {evaluation_results}")

if __name__ == "__main__":
    main()
