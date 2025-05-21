#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trainer for the ML Provenance Tracker Framework.
This module handles model training and evaluation.
"""

import os
import time
import logging
import datetime
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.tof_ml.data.data_manager import DataManager
from src.tof_ml.data.data_provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles model training, evaluation, and checkpoint management.
    
    This class is responsible for:
    - Training models
    - Evaluating model performance
    - Managing checkpoints and early stopping
    - Tracking training metrics
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: Any,
        data_manager: DataManager,
        output_dir: str,
        provenance_tracker: Optional[ProvenanceTracker] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            model: The model to train
            data_manager: Data manager instance with train/val/test data
            output_dir: Directory to save model checkpoints and outputs
            provenance_tracker: Optional provenance tracker instance
        """
        self.config = config
        self.model = model
        self.data_manager = data_manager
        self.output_dir = output_dir
        self.provenance_tracker = provenance_tracker
        
        # Training configuration
        self.model_config = config.get("model", {})
        self.epochs = self.model_config.get("epochs", 10)
        self.batch_size = self.model_config.get("batch_size", 32)
        self.early_stopping = self.model_config.get("early_stopping", {})
        self.checkpoint_config = self.model_config.get("checkpoint", {})
        
        # Initialize training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.training_history = []
        self.early_stop_counter = 0
        self.early_stop_patience = self.early_stopping.get("patience", 10)
        self.test_metrics = None
        
        # Initialize metadata
        self.meta_data = {
            "trainer_init_time": datetime.datetime.now().isoformat(),
            "training_config": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "early_stopping": self.early_stopping,
                "checkpoint": self.checkpoint_config
            }
        }
        
        logger.info("Trainer initialized")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dictionary of training results
        """
        logger.info("Starting model training")
        
        # Get training and validation data
        X_train, y_train = self.data_manager.get_train_data()
        X_val, y_val = self.data_manager.get_val_data()
        
        # Record start time
        start_time = time.time()

        # Initialize model hash ID for provenance tracking
        model_hash = 0
        if self.provenance_tracker:
            import hashlib
            model_hash = hashlib.md5(str(self.model).encode()).hexdigest()
        
        # Set up checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up callbacks
        callbacks = self._setup_callbacks()
        
        # Train the model
        if hasattr(self.model, "fit"):
            # Keras-like model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            # Store training history
            self.training_history = history.history
        
        # Record end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "final_model")
        self._save_model(final_model_path)
        
        # Update metadata
        self.meta_data.update({
            "training_start": datetime.datetime.fromtimestamp(start_time).isoformat(),
            "training_end": datetime.datetime.fromtimestamp(end_time).isoformat(),
            "training_time_seconds": training_time,
            "epochs_completed": self.current_epoch,
            "early_stopped": self.current_epoch < self.epochs,
            "best_val_loss": self.best_val_loss,
            "best_model_path": self.best_model_path,
            "final_model_path": final_model_path
        })
        
        # Record in provenance tracker
        if self.provenance_tracker:
            # Get data IDs from data manager
            train_data_id = self.data_manager.metadata["splitting"].get("train_hash")
            val_data_id = self.data_manager.metadata["splitting"].get("val_hash")
            
            self.provenance_tracker.record_model_training(
                model_id=model_hash,
                train_data_id=train_data_id,
                val_data_id=val_data_id,
                model_info=self.model_config,
                training_info=self.meta_data
            )
        
        # Return training results
        results = {
            "epochs_completed": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "training_time": training_time,
            "best_model_path": self.best_model_path,
            "final_model_path": final_model_path,
            "training_history": self.training_history
        }
        
        logger.info(f"Model training completed in {training_time:.2f} seconds with best val_loss: {self.best_val_loss:.4f}")
        return results

    def evaluate(self, model=None) -> Dict[str, Any]:
        """
        Evaluate a model on the test dataset.

        Args:
            model: Optional model to evaluate (defaults to self.model)

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test data")

        if model is None:
            model = self.model

        # Get test data
        X_test, y_test = self.data_manager.get_test_data()

        # Record start time
        start_time = time.time()

        # Make predictions
        y_pred = model.predict(X_test)

        # Initialize metrics dictionary
        metrics = {}

        # Core evaluation - prioritize the model's own evaluation methods
        if hasattr(model, "evaluate"):
            # Get basic metrics from model's evaluate method
            eval_result = model.evaluate(X_test, y_test, verbose=0)

            # Handle different return types from evaluate
            if isinstance(eval_result, dict):
                # If model returns a dictionary, use it directly
                metrics.update(eval_result)
                test_loss = eval_result.get("test_loss", 0.0)
            elif isinstance(eval_result, list) and hasattr(model, "metrics_names"):
                # Handle Keras-style evaluate that returns a list
                for i, metric_name in enumerate(model.metrics_names):
                    metrics[f"test_{metric_name}"] = eval_result[i]
                test_loss = eval_result[0]  # Usually the first item is loss
            else:
                # Single scalar value (common in sklearn)
                metrics["test_loss"] = eval_result
                test_loss = eval_result
        else:
            # Initialize test_loss placeholder
            test_loss = 0.0

        # Get custom metrics directly from the model if available
        if hasattr(model, "custom_metrics"):
            custom_metrics = model.custom_metrics(y_test, y_pred)
            metrics.update(custom_metrics)

            # Use model's loss value function if available
            if hasattr(model, "get_loss_value") and "test_loss" not in metrics:
                test_loss = model.get_loss_value(metrics)
                metrics["test_loss"] = test_loss

        # Record end time
        end_time = time.time()
        eval_time = end_time - start_time

        # Update metadata
        self.meta_data.update({
            "evaluation_time": eval_time,
            "test_loss": test_loss,
            "test_metrics": metrics
        })

        # Record in provenance tracker
        if self.provenance_tracker:
            import hashlib
            model_hash = hashlib.md5(str(model).encode()).hexdigest()
            test_data_id = self.data_manager.metadata["splitting"].get("test_hash")

            self.provenance_tracker.record_model_evaluation(
                model_id=model_hash,
                test_data_id=test_data_id,
                evaluation_results=metrics
            )

        # Store test metrics
        self.test_metrics = metrics

        # Add predictions and true values to results for plotting
        metrics["y_pred"] = y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
        metrics["y_true"] = y_test.tolist() if isinstance(y_test, np.ndarray) else y_test

        logger.info(f"Model evaluation completed in {eval_time:.2f} seconds with test_loss: {test_loss:.4f}")
        return metrics
    
    def _setup_callbacks(self) -> List[Any]:
        """Set up training callbacks."""
        callbacks = []
        
        # Early stopping callback
        if self.early_stopping.get("enabled", True):
            callbacks.append(self._create_early_stopping_callback())
        
        # Checkpoint callback
        if self.checkpoint_config.get("enabled", True):
            callbacks.append(self._create_checkpoint_callback())
        
        # Optional custom callbacks
        custom_callbacks = self.model_config.get("callbacks", [])
        for callback_config in custom_callbacks:
            callback = self._create_custom_callback(callback_config)
            if callback:
                callbacks.append(callback)
        
        return callbacks
    
    def _create_early_stopping_callback(self) -> Any:
        """Create early stopping callback based on framework.
        Right now this just uses Tensorflow early stopping but is extensible"""
        return EarlyStopping(
            monitor=self.early_stopping.get("monitor", "val_loss"),
            patience=self.early_stopping.get("patience", 10),
            restore_best_weights=True,
            verbose=1
        )
    
    def _create_checkpoint_callback(self) -> Any:
        """Create model checkpoint callback based on framework."""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}"),
            monitor=self.checkpoint_config.get("monitor", "val_loss"),
            save_best_only=self.checkpoint_config.get("save_best_only", True),
            verbose=1
        )

    def _create_custom_callback(self, callback_config: Dict[str, Any]) -> Optional[Any]:
        """Create a custom callback from configuration."""
        callback_type = callback_config.get("type")
        
        if not callback_type:
            logger.warning(f"Callback type not specified in config: {callback_config}")
            return None
        
        # Try to import the callback class
        try:
            module_name = callback_config.get("module", "tensorflow.keras.callbacks")
            callback_class_name = callback_config.get("class", callback_type)
            
            module = __import__(module_name, fromlist=[callback_class_name])
            callback_class = getattr(module, callback_class_name)
            
            # Initialize with parameters
            params = {k: v for k, v in callback_config.items() if k not in ["type", "module", "class"]}
            return callback_class(**params)
        
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to create callback {callback_type}: {e}")
            return None
    
    def _save_model(self, path: str) -> str:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            The full path to the saved model
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model based on its type
        if hasattr(self.model, "save"):
            # Keras-like model
            self.model.save(path)
        else:
            # Fallback to pickle
            import pickle
            with open(f"{path}.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            path = f"{path}.pkl"
        
        logger.info(f"Model saved to {path}")
        return path
    
    def get_learning_curves(self) -> Dict[str, List[float]]:
        """
        Get learning curves from training history.
        
        Returns:
            Dictionary of learning curves
        """
        if not self.training_history:
            logger.warning("No training history available")
            return {}
        
        # Extract metrics from history
        metrics = {}
        for key in self.training_history[0].keys():
            metrics[key] = [epoch[key] for epoch in self.training_history]
        
        return metrics
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get training metadata."""
        return self.meta_data
