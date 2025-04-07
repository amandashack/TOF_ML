#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration utilities for the ML Provenance Tracker Framework.
This module provides functions for handling configurations.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override configuration into base configuration.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for required fields and values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid, raises exception otherwise
    """
    # Check required sections
    required_sections = ["data", "model", "output_dir"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Check data section
    if "data" in config:
        data_config = config["data"]
        if "loader_config_key" not in data_config:
            raise ValueError("Missing required field: data.loader_config_key")
    
    # Check model section
    if "model" in config:
        model_config = config["model"]
        if "type" not in model_config:
            raise ValueError("Missing required field: model.type")
    
    # Ensure output directory can be created
    output_dir = config.get("output_dir")
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create output directory: {output_dir}, error: {e}")
    
    logger.info("Configuration validated successfully")
    return True

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not config:
        logger.warning(f"Empty configuration loaded from {config_path}")
        config = {}
    
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {config_path}")

def generate_default_config() -> Dict[str, Any]:
    """
    Generate default configuration.
    
    Returns:
        Default configuration dictionary
    """
    default_config = {
        "experiment_name": "default_experiment",
        "output_dir": "./output",
        "provenance": {
            "enabled": True,
            "db_path": "./provenance"
        },
        "class_mapping_path": "config/class_mapping_config.yaml",
        "data": {
            "loader_config_key": "CSVLoader",
            "dataset_path": "./data/sample_data.csv",
            "n_samples": None,
            "feature_columns": [],
            "target_columns": []
        },
        "preprocessing": {
            "transformers": [
                {
                    "type": "Normalizer",
                    "columns": [],
                    "method": "standard"
                }
            ]
        },
        "data_splitting": {
            "type": "RandomSplitter",
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42
        },
        "model": {
            "type": "MLPKerasRegressor",
            "hidden_layers": [32, 32],
            "activations": ["relu", "relu"],
            "learning_rate": 0.001,
            "optimizer_name": "Adam",
            "epochs": 50,
            "batch_size": 32,
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "monitor": "val_loss"
            },
            "checkpoint": {
                "enabled": True,
                "save_best_only": True,
                "monitor": "val_loss"
            }
        },
        "reporting": {
            "enabled_reports": ["data", "training", "evaluation", "summary"],
            "plots": {
                "dpi": 300,
                "style": "seaborn-darkgrid"
            }
        },
        "use_database": True,
        "database": {
            "config_path": "config/database_config.yaml"
        }
    }
    
    return default_config
