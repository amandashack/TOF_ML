#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating the MNIST plugin for the ML Provenance Tracker Framework.
"""

import os
import sys
import logging
import yaml
from typing import Dict, Any

from src.tof_ml.pipeline.orchestrator import PipelineOrchestrator
from src.tof_ml.pipeline.plugins.registry import get_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_mnist_example():
    """Run the MNIST example using the plugin architecture."""
    # Load MNIST configuration
    config_path = "config/mnist_config.yaml"
    config = load_config(config_path)
    
    # Display available plugins
    registry = get_registry()
    
    print("\nAvailable plugins:")
    print("Data Loaders:", registry.list_data_loaders())
    print("Models:", registry.list_models())
    print("Report Generators:", registry.list_report_generators())
    
    # Initialize and run pipeline
    orchestrator = PipelineOrchestrator(config)
    
    # Run the complete pipeline
    logger.info("Running MNIST pipeline")
    results = orchestrator.run_pipeline()
    
    # Print output directory
    output_dir = orchestrator.output_dir
    print(f"\nExecution completed successfully. Results saved to:\n{output_dir}\n")
    
    # Print key metrics
    if "evaluation_results" in results:
        print("\nEvaluation Results:")
        for key, value in results["evaluation_results"].items():
            if key not in ["y_true", "y_pred"]:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    run_mnist_example()
