#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command line interface for the ML Provenance Tracker Framework.
This module provides the entry point for running ML pipelines with full provenance tracking.
"""

import argparse
import os
import yaml
import logging
from typing import Dict, Any, Optional

from tof_ml.pipeline.orchestrator import PipelineOrchestrator
from tof_ml.utils.config_utils import merge_configs, validate_config

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ML Pipeline with Provenance Tracking")
    
    # Main operation modes
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "report", "full"], 
                        default="full", help="Pipeline operation mode")
    
    # Configuration options
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                        help="Path to base configuration file")
    parser.add_argument("--override", type=str, default=None,
                        help="Path to override configuration file")
    
    # Data options
    parser.add_argument("--data-loader", type=str, 
                        help="Data loader to use (overrides config)")
    parser.add_argument("--dataset-path", type=str,
                        help="Path to dataset (overrides config)")
    parser.add_argument("--n-samples", type=int,
                        help="Number of samples to use (overrides config)")
    
    # Model options
    parser.add_argument("--model-type", type=str,
                        help="Model type to use (overrides config)")
    parser.add_argument("--model-params", type=str,
                        help="JSON string of model parameters (overrides config)")
    parser.add_argument("--load-model", type=str,
                        help="Path to load a pretrained model")
    
    # Training options
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int,
                        help="Batch size (overrides config)")
    parser.add_argument("--learning-rate", type=float,
                        help="Learning rate (overrides config)")
    
    # Output options
    parser.add_argument("--output-dir", type=str,
                        help="Output directory (overrides config)")
    parser.add_argument("--experiment-name", type=str,
                        help="Experiment name (overrides config)")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    
    # Database options
    parser.add_argument("--db-config", type=str,
                        help="Path to database configuration (overrides config)")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip database logging")
    
    return parser.parse_args()

def setup_logging(log_level: str, output_dir: Optional[str] = None):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "pipeline.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

def build_config_from_args(args, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build configuration dictionary from command line arguments."""
    config = base_config.copy()
    
    # Override with args if provided
    if args.data_loader:
        config["data"]["loader_config_key"] = args.data_loader
    
    if args.dataset_path:
        config["data"]["dataset_path"] = args.dataset_path
        
    if args.n_samples is not None:
        config["data"]["n_samples"] = args.n_samples
        
    if args.model_type:
        config["model"]["type"] = args.model_type
        
    if args.model_params:
        import json
        model_params = json.loads(args.model_params)
        config["model"].update(model_params)
        
    if args.epochs:
        config["model"]["epochs"] = args.epochs
        
    if args.batch_size:
        config["model"]["batch_size"] = args.batch_size
        
    if args.learning_rate:
        config["model"]["learning_rate"] = args.learning_rate
        
    if args.output_dir:
        config["output_dir"] = args.output_dir
        
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
        
    if args.skip_db:
        config["use_database"] = False
        
    # Add additional CLI args as metadata
    config["metadata"] = config.get("metadata", {})
    config["metadata"]["cli_args"] = vars(args)
    
    return config

def main():
    """Main entry point for the ML pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load override configuration if provided
    if args.override:
        with open(args.override, 'r') as f:
            override_config = yaml.safe_load(f)
        base_config = merge_configs(base_config, override_config)
    
    # Set up output directory
    output_dir = args.output_dir or base_config.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(args.log_level, output_dir)
    
    # Build final configuration
    config = build_config_from_args(args, base_config)
    
    # Validate configuration
    validate_config(config)
    
    # Initialize and run pipeline
    pipeline = PipelineOrchestrator(config)
    
    if args.mode == "train":
        pipeline.run_training()
    elif args.mode == "evaluate":
        pipeline.run_evaluation()
    elif args.mode == "report":
        pipeline.generate_report()
    else:  # full
        pipeline.run_pipeline()
        
    logger.info(f"Pipeline completed successfully. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
