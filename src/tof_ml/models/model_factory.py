#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Factory for the ML Provenance Tracker Framework.
This module handles creating and loading models.
"""

import os
import logging
import importlib
import datetime
from typing import Dict, Any, Optional, Union, Type

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating and loading machine learning models.
    
    This class handles:
    - Model instantiation based on configuration
    - Loading pretrained models
    - Saving models
    - Model versioning and metadata
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.class_mapping = self._load_class_mapping()
        self.metadata = {}
        
        logger.info("ModelFactory initialized")
    
    def _load_class_mapping(self) -> Dict[str, Dict[str, str]]:
        """Load class mapping from configuration."""
        mapping_path = self.config.get("class_mapping_path", "config/class_mapping_config.yaml")
        
        import yaml
        with open(mapping_path, 'r') as f:
            class_mapping = yaml.safe_load(f)
        
        return class_mapping
    
    def create_model(self) -> Any:
        """
        Create a model based on the configuration.
        
        Returns:
            The created model instance
        """
        logger.info("Creating model")
        
        model_type = self.model_config.get("type")
        if not model_type:
            raise ValueError("Model type not specified in configuration")
        
        # Get model class from mapping
        model_mapping = self.class_mapping.get("Model", {})
        model_class_str = model_mapping.get(model_type)
        
        if not model_class_str:
            raise ValueError(f"Model class not found for type: {model_type}")
        
        # Dynamically import model class
        module_name, class_name = model_class_str.rsplit('.', 1)
        model_module = importlib.import_module(module_name)
        model_class = getattr(model_module, class_name)
        
        # Extract model-specific parameters from config
        model_params = self.model_config.copy()
        
        # Remove non-model parameters
        for key in ["type", "epochs", "batch_size", "early_stopping", "checkpoint"]:
            model_params.pop(key, None)
        
        # Create model instance
        model = model_class(**model_params)
        
        # Store metadata
        self.metadata = {
            "model_type": model_type,
            "model_class": model_class_str,
            "model_parameters": model_params,
            "creation_time": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Model created: {model_type} ({model_class_str})")
        return model
    
    def save_model(self, model: Any, output_path: str) -> str:
        """
        Save a model to disk.
        
        Args:
            model: The model to save
            output_path: Path to save the model
            
        Returns:
            The full path to the saved model
        """
        logger.info(f"Saving model to {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get the model's save method
        if hasattr(model, "save"):
            # Use the model's save method
            model.save(output_path)
        else:
            # Use pickle as a fallback
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata alongside the model
        metadata_path = f"{output_path}.metadata.json"
        self._save_metadata(metadata_path)
        
        logger.info(f"Model saved to {output_path}")
        return output_path
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a pretrained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            The loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try to load metadata if available
        metadata_path = f"{model_path}.metadata.json"
        if os.path.exists(metadata_path):
            self._load_metadata(metadata_path)
            
            # Use model class from metadata if available
            model_class_str = self.metadata.get("model_class")
            if model_class_str:
                module_name, class_name = model_class_str.rsplit('.', 1)
                model_module = importlib.import_module(module_name)
                model_class = getattr(model_module, class_name)
                
                # Check if the class has a load method
                if hasattr(model_class, "load"):
                    model = model_class.load(model_path)
                    logger.info(f"Model loaded using {model_class.__name__}.load()")
                    return model
        
        # Try different loading methods
        model = self._try_different_loading_methods(model_path)
        
        if model is None:
            raise ValueError(f"Failed to load model from {model_path}")
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def _try_different_loading_methods(self, model_path: str) -> Optional[Any]:
        """Try different methods to load a model."""
        # Try loading with pickle
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                logger.info("Model loaded using pickle")
                return model
        except Exception as e:
            logger.debug(f"Failed to load model with pickle: {e}")
        
        # Try loading with joblib
        try:
            import joblib
            model = joblib.load(model_path)
            logger.info("Model loaded using joblib")
            return model
        except Exception as e:
            logger.debug(f"Failed to load model with joblib: {e}")
        
        # Try loading with TensorFlow if it's a saved model
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded using TensorFlow")
            return model
        except Exception as e:
            logger.debug(f"Failed to load model with TensorFlow: {e}")
        
        # Try loading with PyTorch if it's a saved state dict
        try:
            import torch
            model_state = torch.load(model_path)
            logger.info("Model state loaded using PyTorch")
            return model_state
        except Exception as e:
            logger.debug(f"Failed to load model state with PyTorch: {e}")
        
        return None
    
    def _save_metadata(self, metadata_path: str) -> None:
        """Save model metadata to disk."""
        import json
        
        # Add timestamp and version
        self.metadata["saved_time"] = datetime.datetime.now().isoformat()
        self.metadata["version"] = self.metadata.get("version", "1.0")
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _load_metadata(self, metadata_path: str) -> None:
        """Load model metadata from disk."""
        import json
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the model."""
        return self.metadata
