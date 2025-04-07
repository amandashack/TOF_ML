#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for preprocessing ToF data using the ML Provenance Tracker Framework.

This script demonstrates how to:
1. Load raw data
2. Apply preprocessing transformations
3. Split data into train/validation/test sets
4. Save preprocessed data
5. Track provenance throughout the process

Usage:
    python examples/preprocess_data.py --config config/base_config.yaml
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
from src.tof_ml.database.api import DBApi

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess ToF data")
    
    # Configuration options
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data-path", type=str,
                        help="Path to raw data (overrides config)")
    parser.add_argument("--output-dir", type=str,
                        help="Output directory (overrides config)")
    parser.add_argument("--experiment-name", type=str,
                        help="Experiment name (overrides config)")
    
    # Preprocessing options
    parser.add_argument("--normalization-method", type=str, choices=["standard", "minmax", "robust"],
                        help="Normalization method (overrides config)")
    parser.add_argument("--log-transform", action="store_true",
                        help="Apply log transformation to ToF values")
    parser.add_argument("--n-samples", type=int,
                        help="Number of samples to use (overrides config)")
    
    # Splitting options
    parser.add_argument("--test-size", type=float,
                        help="Test set size ratio (overrides config)")
    parser.add_argument("--val-size", type=float,
                        help="Validation set size ratio (overrides config)")
    parser.add_argument("--random-seed", type=int,
                        help="Random seed for splitting (overrides config)")
    
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
            logging.FileHandler("data_preprocessing.log")
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
    
    # Output options
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    
    # Preprocessing options
    if args.normalization_method:
        for i, transformer in enumerate(config["preprocessing"]["transformers"]):
            if transformer.get("type") == "Normalizer":
                config["preprocessing"]["transformers"][i]["method"] = args.normalization_method
    
    if args.log_transform:
        # Add log transform if not already present
        has_log_transform = any(transformer.get("type") == "LogTransformer" 
                              for transformer in config["preprocessing"]["transformers"])
        if not has_log_transform:
            config["preprocessing"]["transformers"].append({
                "type": "LogTransformer",
                "columns": ["tof"],
                "base": 10,
                "offset": 1e-6,
                "handle_negatives": "offset"
            })
    
    # Splitting options
    if args.test_size:
        config["data_splitting"]["test_size"] = args.test_size
    
    if args.val_size:
        config["data_splitting"]["val_size"] = args.val_size
    
    if args.random_seed:
        config["data_splitting"]["random_state"] = args.random_seed
    
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
    config = update_config_from_args(config, args)
    
    # Set up output directory
    output_dir = config.get("output_dir", "./output")
    experiment_name = config.get("experiment_name", "data_preprocessing")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Update config with experiment directory
    config["output_dir"] = experiment_dir
    
    # Initialize provenance tracker
    provenance_tracker = ProvenanceTracker(config)
    
    # Initialize data manager
    data_manager = DataManager(
        config=config,
        output_dir=os.path.join(experiment_dir, "data"),
        provenance_tracker=provenance_tracker
    )
    
    # Load raw data
    logging.info("Loading raw data...")
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
    
    # Visualize provenance graph
    provenance_tracker.visualize_provenance_graph(
        os.path.join(experiment_dir, "provenance", "lineage_graph.svg")
    )
    
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
            "total_samples": data_manager.get_metadata().get("total_samples", 0)
        }
        
        # Record preprocessing run
        experiment_id = db_api.record_experiment(
            metadata=metadata,
            config=config,
            report_paths=saved_paths
        )
        
        logging.info(f"Preprocessing run recorded in database with ID: {experiment_id}")
    
    logging.info(f"Data preprocessing completed. Results saved to {experiment_dir}")
    logging.info(f"Saved datasets: {list(saved_paths.keys())}")
    logging.info(f"Provenance graph: {provenance_path}")

if __name__ == "__main__":
    main()
