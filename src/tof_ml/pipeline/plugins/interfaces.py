#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plugin interfaces for the ML Provenance Tracker Framework.
This module provides abstract base classes for all plugin components.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union


class DataLoaderPlugin(ABC):
    """Abstract base class for data loader plugins."""

    @abstractmethod
    def load_data(self) -> np.ndarray:
        """
        Load data from the data source.

        Returns:
            np.ndarray: Loaded data with shape (n_samples, n_features)
        """
        pass

    @abstractmethod
    def extract_features_and_targets(self, data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature and target arrays from data.

        Args:
            data: Optional data array. If None, uses loaded data.

        Returns:
            Tuple of (features, targets)
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loader and loaded data.

        Returns:
            Dictionary containing loader metadata
        """
        pass


class ModelPlugin(ABC):
    """Abstract base class for model plugins."""

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Train the model.
        
        Args:
            X: Feature data
            y: Target data
            **kwargs: Additional training parameters
            
        Returns:
            Training history or other training result
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the model.
        
        Args:
            X: Feature data
            
        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def evaluate(self, X, y, **kwargs):
        """
        Evaluate the model.
        
        Args:
            X: Feature data
            y: Target data
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def custom_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate custom evaluation metrics.
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            
        Returns:
            Dictionary of custom metrics
        """
        pass


class ReportGeneratorPlugin(ABC):
    """Abstract base class for report generator plugins."""

    @abstractmethod
    def generate_data_report(self) -> str:
        """
        Generate a report on the dataset.
        
        Returns:
            Path to the generated report
        """
        pass

    @abstractmethod
    def generate_training_report(self) -> str:
        """
        Generate a report on model training.
        
        Returns:
            Path to the generated report
        """
        pass

    @abstractmethod
    def generate_evaluation_report(self) -> str:
        """
        Generate a report on model evaluation.
        
        Returns:
            Path to the generated report
        """
        pass
    
    @abstractmethod
    def generate_reports(self) -> Dict[str, str]:
        """
        Generate all reports.
        
        Returns:
            Dictionary of report types and their paths
        """
        pass