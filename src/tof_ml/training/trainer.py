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
            # If model has a fit method (sklearn-like or keras-like)
            if self._is_keras_like():
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
            else:
                # Sklearn-like model
                self.model.fit(X_train, y_train)
                # Evaluate on validation set
                val_score = self._evaluate_sklearn(X_val, y_val)
                self.training_history = [{"val_loss": val_score}]
        else:
            # Custom training loop
            self.training_history = self._custom_training_loop(
                X_train, y_train, X_val, y_val, callbacks
            )
        
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
        
        # Evaluate the model
        if hasattr(model, "evaluate"):
            # Keras-like model
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            if isinstance(test_loss, list):
                metrics = {f"test_{name}": value for name, value in zip(model.metrics_names, test_loss)}
                test_loss = metrics.get("test_loss", test_loss[0])
            else:
                metrics = {"test_loss": test_loss}
        elif hasattr(model, "score"):
            # Sklearn-like model
            test_score = model.score(X_test, y_test)
            metrics = {"test_score": test_score}
            test_loss = -test_score  # Negative score as loss
        else:
            # Custom evaluation
            metrics = self._custom_evaluation(model, X_test, y_test)
            test_loss = metrics.get("test_loss", 0.0)
        
        # Make predictions
        y_pred = self._get_predictions(model, X_test)
        
        # Calculate additional metrics
        additional_metrics = self._calculate_additional_metrics(y_test, y_pred)
        metrics.update(additional_metrics)
        
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
        
        # Custom callbacks
        custom_callbacks = self.model_config.get("callbacks", [])
        for callback_config in custom_callbacks:
            callback = self._create_custom_callback(callback_config)
            if callback:
                callbacks.append(callback)
        
        return callbacks
    
    def _create_early_stopping_callback(self) -> Any:
        """Create early stopping callback based on framework."""
        if self._is_keras_like():
            try:
                from tensorflow.keras.callbacks import EarlyStopping
                
                return EarlyStopping(
                    monitor=self.early_stopping.get("monitor", "val_loss"),
                    patience=self.early_stopping.get("patience", 10),
                    restore_best_weights=True,
                    verbose=1
                )
            except ImportError:
                logger.warning("TensorFlow not available, using custom early stopping")
        
        # Return a custom early stopping object
        return {"type": "early_stopping"}
    
    def _create_checkpoint_callback(self) -> Any:
        """Create model checkpoint callback based on framework."""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if self._is_keras_like():
            try:
                from tensorflow.keras.callbacks import ModelCheckpoint
                
                return ModelCheckpoint(
                    filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}"),
                    monitor=self.checkpoint_config.get("monitor", "val_loss"),
                    save_best_only=self.checkpoint_config.get("save_best_only", True),
                    verbose=1
                )
            except ImportError:
                logger.warning("TensorFlow not available, using custom checkpoint")
        
        # Return a custom checkpoint object
        return {"type": "checkpoint", "filepath": checkpoint_dir}
    
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
    
    def _custom_training_loop(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        callbacks: List[Any]
    ) -> List[Dict[str, float]]:
        """
        Custom training loop for models without a standard fit method.
        
        Returns:
            List of dictionaries with training history
        """
        history = []
        
        # Check if callbacks include early stopping
        has_early_stopping = any(isinstance(cb, dict) and cb.get("type") == "early_stopping" for cb in callbacks)
        
        # Training loop
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            logger.info(f"Epoch {self.current_epoch}/{self.epochs}")
            
            # Train for one epoch
            train_metrics = self._train_one_epoch(X_train, y_train)
            
            # Evaluate on validation set
            val_metrics = self._validate_one_epoch(X_val, y_val)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            history.append(epoch_metrics)
            
            # Log metrics
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
            logger.info(f"Epoch {self.current_epoch}/{self.epochs} - {metrics_str}")
            
            # Check for improvement
            val_loss = val_metrics.get("val_loss", float('inf'))
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                
                # Save best model
                best_model_path = os.path.join(self.output_dir, f"best_model_epoch_{self.current_epoch}")
                self.best_model_path = self._save_model(best_model_path)
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.early_stop_counter += 1
            
            # Handle early stopping
            if has_early_stopping and self.early_stop_counter >= self.early_stop_patience:
                logger.info(f"Early stopping triggered after {self.current_epoch} epochs")
                break
        
        return history
    
    def _train_one_epoch(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train for one epoch in custom training loop.
        
        Returns:
            Dictionary of training metrics
        """
        # This is a placeholder for custom model training
        # In a real implementation, you would add model-specific training code here
        
        # For example, if the model has a partial_fit method (like some sklearn models)
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X_train, y_train)
        
        # Return dummy metrics
        return {"loss": 0.0}
    
    def _validate_one_epoch(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Validate model after one epoch in custom training loop.
        
        Returns:
            Dictionary of validation metrics
        """
        # This is a placeholder for custom model validation
        # In a real implementation, you would add model-specific validation code here
        
        # For example, if the model has a score method (like sklearn models)
        if hasattr(self.model, "score"):
            score = self.model.score(X_val, y_val)
            return {"val_loss": -score}  # Negative score as loss
        
        # Placeholder for custom validation
        y_pred = self._get_predictions(self.model, X_val)
        
        # Calculate mean squared error as validation loss
        mse = np.mean((y_val - y_pred) ** 2)
        
        return {"val_loss": mse}
    
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
    
    def _get_predictions(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get predictions from a model."""
        if hasattr(model, "predict"):
            # Standard predict method
            y_pred = model.predict(X)
        else:
            # No standard predict method
            logger.warning("Model does not have a standard predict method")
            y_pred = np.zeros(X.shape[0])
        
        return y_pred
    
    def _calculate_additional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate additional evaluation metrics."""
        metrics = {}
        
        # Reshape if needed
        if len(y_true.shape) > 1 and y_true.shape[1] == 1:
            y_true = y_true.ravel()
        
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        metrics["mse"] = float(mse)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        metrics["rmse"] = float(rmse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        metrics["mae"] = float(mae)
        
        # R^2 Score
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        
        if ss_total > 0:
            r2 = 1 - (ss_residual / ss_total)
            metrics["r2"] = float(r2)
        
        return metrics
    
    def _is_keras_like(self) -> bool:
        """Check if the model is Keras-like."""
        return (
            hasattr(self.model, "fit") and 
            hasattr(self.model, "predict") and 
            hasattr(self.model, "evaluate")
        )
    
    def _evaluate_sklearn(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate an sklearn-like model."""
        if hasattr(self.model, "score"):
            score = self.model.score(X_val, y_val)
            return -score  # Negative score as loss (higher score is better in sklearn)
        else:
            # Use predictions to calculate MSE
            y_pred = self._get_predictions(self.model, X_val)
            return float(np.mean((y_val - y_pred) ** 2))
    
    def _custom_evaluation(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Custom evaluation for models without standard evaluation methods.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = self._get_predictions(model, X_test)
        
        # Calculate metrics
        metrics = self._calculate_additional_metrics(y_test, y_pred)
        
        # Use MSE as test_loss
        metrics["test_loss"] = metrics.get("mse", 0.0)
        
        return metrics
    
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
