#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plugin Registry for the ML Provenance Tracker Framework.
This module provides a registry for managing and instantiating plugins.
"""

import logging
import importlib
from typing import Dict, Any, Type, List, Optional

from src.tof_ml.pipeline.plugins.interfaces import (
    DataLoaderPlugin,
    ModelPlugin,
    ReportGeneratorPlugin
)

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Registry for managing and instantiating plugins.
    
    This class maintains a registry of available plugins for different components
    of the ML pipeline and provides methods to register and retrieve them.
    """
    
    def __init__(self):
        """Initialize the plugin registry."""
        self.data_loaders = {}
        self.models = {}
        self.report_generators = {}
        
        logger.info("PluginRegistry initialized")
    
    def register_data_loader(self, name: str, loader_class: Type[DataLoaderPlugin]):
        """
        Register a data loader plugin.
        
        Args:
            name: Name of the data loader
            loader_class: Class of the data loader plugin
        """
        self.data_loaders[name] = loader_class
        logger.info(f"Registered data loader plugin: {name}")
    
    def register_model(self, name: str, model_class: Type[ModelPlugin]):
        """
        Register a model plugin.
        
        Args:
            name: Name of the model
            model_class: Class of the model plugin
        """
        self.models[name] = model_class
        logger.info(f"Registered model plugin: {name}")
    
    def register_report_generator(self, name: str, generator_class: Type[ReportGeneratorPlugin]):
        """
        Register a report generator plugin.
        
        Args:
            name: Name of the report generator
            generator_class: Class of the report generator plugin
        """
        self.report_generators[name] = generator_class
        logger.info(f"Registered report generator plugin: {name}")
    
    def get_data_loader(self, name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> DataLoaderPlugin:
        """
        Get an instance of a data loader plugin.
        
        Args:
            name: Name of the data loader
            config: Optional configuration dictionary
            **kwargs: Additional parameters to pass to the loader constructor
            
        Returns:
            An instance of the requested data loader plugin
            
        Raises:
            KeyError: If the data loader is not registered
        """
        if name not in self.data_loaders:
            raise KeyError(f"Data loader plugin not found: {name}")
        
        loader_class = self.data_loaders[name]
        return loader_class(config, **kwargs)
    
    def get_model(self, name: str, **kwargs) -> ModelPlugin:
        """
        Get an instance of a model plugin.
        
        Args:
            name: Name of the model
            **kwargs: Additional parameters to pass to the model constructor
            
        Returns:
            An instance of the requested model plugin
            
        Raises:
            KeyError: If the model is not registered
        """
        if name not in self.models:
            raise KeyError(f"Model plugin not found: {name}")
        
        model_class = self.models[name]
        return model_class(**kwargs)
    
    def get_report_generator(self, name: str, config: Dict[str, Any], **kwargs) -> ReportGeneratorPlugin:
        """
        Get an instance of a report generator plugin.
        
        Args:
            name: Name of the report generator
            config: Configuration dictionary
            **kwargs: Additional parameters to pass to the report generator constructor
            
        Returns:
            An instance of the requested report generator plugin
            
        Raises:
            KeyError: If the report generator is not registered
        """
        if name not in self.report_generators:
            raise KeyError(f"Report generator plugin not found: {name}")
        
        generator_class = self.report_generators[name]
        return generator_class(config, **kwargs)
    
    def list_data_loaders(self) -> List[str]:
        """List all registered data loader plugins."""
        return list(self.data_loaders.keys())
    
    def list_models(self) -> List[str]:
        """List all registered model plugins."""
        return list(self.models.keys())
    
    def list_report_generators(self) -> List[str]:
        """List all registered report generator plugins."""
        return list(self.report_generators.keys())
    
    def register_plugins_from_config(self, class_mapping: Dict[str, Dict[str, str]]):
        """
        Register plugins from a class mapping configuration.
        
        Args:
            class_mapping: Dictionary mapping plugin types and names to class paths
        """
        # Register data loaders
        for name, class_path in class_mapping.get("Loader", {}).items():
            try:
                module_name, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                loader_class = getattr(module, class_name)
                self.register_data_loader(name, loader_class)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to register data loader {name}: {e}")
        
        # Register models
        for name, class_path in class_mapping.get("Model", {}).items():
            try:
                module_name, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
                self.register_model(name, model_class)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to register model {name}: {e}")
        
        # Register report generators
        for name, class_path in class_mapping.get("ReportGenerator", {}).items():
            try:
                module_name, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                generator_class = getattr(module, class_name)
                self.register_report_generator(name, generator_class)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to register report generator {name}: {e}")
        
        logger.info("Registered plugins from config")


# Singleton instance of the plugin registry
registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get the singleton instance of the plugin registry."""
    return registry