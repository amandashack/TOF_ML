#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for training a ToF to Energy model using the ML Provenance Tracker Framework.

This script demonstrates how to:
1. Load preprocessed data
2. Create and train a ToF to Energy model
3. Evaluate the model
4. Generate reports
5. Track provenance throughout the process

Usage:
    python examples/train_tof_model.py --config config/tof_model_config.yaml
"""

import os
import sys
import argparse
import logging
import yaml
from typing import Dict, Any

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tof_ml.data.data_manager import DataManager
from src.tof_ml.data.data_provenance import ProvenanceTracker
from src.tof_ml.models.model_factory import ModelFactory
from src.tof_ml.training.trainer import Trainer
from src.tof_ml.reporting.report_generator import ReportGenerator
from src.tof_ml.database.api import DBApi

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ToF to Energy model")
    
    # Configuration options
    parser.add_argument("--config", type=str, default="config/tof_model_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data-path", type=str,
                        help="Path to preprocessed data (overrides config)")
    parser.add_argument("--output-dir", type=str,
                        help="Output directory (overrides config)")
    parser.add_argument("--experiment-name", type=str,
                        help="Experiment name (overrides config)")
    
    # Model options
    parser.add_argument("--hidden-layers", type=str,
                        help="Comma-separated list of hidden layer sizes (overrides config)")
    parser.add_argument("--activations", type=str,
                        help="Comma-separated list of activation functions (overrides config)")
    parser.add_argument("--learning-rate", type=float,
                        help="Learning rate (overrides config)")
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int,
                        help="Batch size (overrides config)")
    
    # Database options
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip database logging")
    
    # Reporting options
    parser.add_argument("--skip-reports", action="store_true",
                        help="Skip report generation")
    
    return parser.parse_args()

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("tof_model_training.log")
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
    
    # Output options
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    
    # Model options
    if args.hidden_layers:
        layers = [int(layer) for layer in args.hidden_layers.split(",")]
        config["model"]["hidden_layers"] = layers
    
    if args.activations:
        activations = args.activations.split(",")
        config["model"]["activations"] = activations
    
    if args.learning_rate:
        config["model"]["learning_rate"] = args.learning_rate
    
    if args.epochs:
        config["model"]["epochs"] = args.epochs
    
    if args.batch_size:
        config["model"]["batch_size"] = args.batch_size
    
    # Database options
    if args.skip_db:
        config["use_database"] = False
    
    # Reporting options
    if args.skip_reports:
        config["reporting"]["enabled_reports"] = []
    
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
    config = update_config_from_args(config, args)
    
    # Set up output directory
    output_dir = config.get("output_dir", "./output")
    experiment_name = config.get("experiment_name", "tof_model")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Update config with experiment directory
    config["output_dir"] = experiment_dir
    
    # Initialize provenance tracker
    provenance_tracker = ProvenanceTracker(config)
    
    # Initialize data manager and load preprocessed data
    data_manager = DataManager(
        config=config,
        output_dir=os.path.join(experiment_dir, "data"),
        provenance_tracker=provenance_tracker
    )
    
    # Load preprocessed data
    logging.info("Loading preprocessed data...")
    data_path = config["data"]["dataset_path"]
    if os.path.isdir(data_path):
        # Load from directory with multiple files
        datasets = {
            "train_data": os.path.join(data_path, "train_data.pkl"),
            "val_data": os.path.join(data_path, "val_data.pkl"),
            "test_data": os.path.join(data_path, "test_data.pkl"),
            "metadata": os.path.join(data_path, "data_metadata.json")
        }
        data_manager.load_datasets(datasets)
    else:
        # Load from single file and split
        data_manager.load_data()
        data_manager.split_data()
    
    # Initialize model factory
    model_factory = ModelFactory(config)
    
    # Create model
    logging.info("Creating model...")
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
    logging.info("Training model...")
    training_results = trainer.train()
    
    # Evaluate model
    logging.info("Evaluating model...")
    evaluation_results = trainer.evaluate()
    
    # Generate reports
    if not args.skip_reports:
        logging.info("Generating reports...")
        report_generator = ReportGenerator(
            config=config,
            data_manager=data_manager,
            training_results=training_results,
            evaluation_results=evaluation_results,
            output_dir=os.path.join(experiment_dir, "reports")
        )
        report_paths = report_generator.generate_reports()
    else:
        report_paths = {}
    
    # Save to database
    if config.get("use_database", True):
        logging.info("Saving to database...")
        db_api = DBApi(config_path=config["database"]["config_path"])
        db_api.record_model_run(
            config_dict=config,
            training_results=training_results,
            model_path=training_results["best_model_path"],
            plot_paths=report_paths
        )
    
    # Save provenance graph
    provenance_path = provenance_tracker.save_provenance_graph(
        os.path.join(experiment_dir, "provenance", "lineage_graph.json")
    )
    
    # Visualize provenance graph
    provenance_tracker.visualize_provenance_graph(
        os.path.join(experiment_dir, "provenance", "lineage_graph.svg")
    )
    
    logging.info(f"Model training completed. Results saved to {experiment_dir}")
    logging.info(f"Best model: {training_results['best_model_path']}")
    logging.info(f"Test metrics: {evaluation_results}")
    logging.info(f"Provenance graph: {provenance_path}")

if __name__ == "__main__":
    main()
