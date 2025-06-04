#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Trainer with explicit attribute tracking and real-time metrics.
"""

import os
import time
import logging
import datetime
import numpy as np
import json
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class TrainingMetricsCallback:
    """Real-time training metrics tracking callback."""

    def __init__(self, trainer, provenance_tracker):
        import tensorflow as tf

        class TFCallback(tf.keras.callbacks.Callback):
            def __init__(self, trainer, provenance_tracker):
                super().__init__()
                self.trainer = trainer
                self.provenance_tracker = provenance_tracker
                self.epoch_start_time = None

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                if not logs:
                    return

                epoch_duration = time.time() - self.epoch_start_time

                # Store epoch metrics in database immediately
                epoch_metrics = {
                    **logs,  # All training metrics (loss, accuracy, val_loss, val_accuracy)
                    "epoch_duration": epoch_duration,
                    "learning_rate": float(self.model.optimizer.learning_rate) if hasattr(self.model.optimizer,
                                                                                          'learning_rate') else 0.0
                }

                # Record metrics to database
                self.provenance_tracker.db_api.store_metrics(
                    run_id=self.provenance_tracker.run_id,
                    metrics=epoch_metrics,
                    stage="training",
                    epoch=epoch + 1
                )

                # Update trainer's best validation loss tracking
                val_loss = logs.get('val_loss')
                if val_loss is not None:
                    if val_loss < self.trainer.best_val_loss:
                        self.trainer.best_val_loss = val_loss
                        self.trainer.best_epoch = epoch + 1
                        logger.info(f"New best val_loss: {val_loss:.6f} at epoch {epoch + 1}")

                        # Record best model milestone
                        self.provenance_tracker.db_api.store_metrics(
                            run_id=self.provenance_tracker.run_id,
                            metrics={"best_val_loss_milestone": val_loss, "best_epoch": epoch + 1},
                            stage="training",
                            epoch=epoch + 1
                        )

                # Log progress
                logger.info(f"Epoch {epoch + 1}: " +
                            ", ".join([f"{k}={v:.4f}" for k, v in logs.items()]))

        self.callback = TFCallback(trainer, provenance_tracker)


class Trainer:
    """Enhanced trainer with comprehensive explicit attribute tracking."""

    def __init__(
            self,
            config: Dict[str, Any],
            model: Any,
            data_manager,
            output_dir: str,
            provenance_tracker
    ):
        """Initialize the enhanced trainer."""
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

        # Explicit training attributes instead of metadata
        self.training_start_time = None
        self.training_end_time = None
        self.training_duration = 0.0
        self.epochs_completed = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.final_train_loss = 0.0
        self.final_val_loss = 0.0
        self.early_stopped = False
        self.model_saved_path = None
        self.best_model_saved_path = None
        self.training_history = {}

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

        logger.info("Enhanced Trainer initialized")

    def train(self) -> Dict[str, Any]:
        """Train the model with comprehensive tracking."""
        logger.info("Starting model training")

        # Get training and validation data
        X_train, y_train = self.data_manager.get_train_data()
        X_val, y_val = self.data_manager.get_val_data()

        # Training parameters
        training_params = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "model_config": self.model_config,
            "train_samples": len(X_train),
            "val_samples": len(X_val)
        }

        with self.provenance_tracker.track_stage("training", training_params):
            # Record start time
            self.training_start_time = time.time()

            # Set up callbacks with artifact tracking
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

            # Record end time and set explicit attributes
            self.training_end_time = time.time()
            self.training_duration = self.training_end_time - self.training_start_time
            self.epochs_completed = len(self.training_history.get('loss', []))

            if self.training_history:
                self.final_train_loss = self.training_history.get('loss', [0])[-1]
                self.final_val_loss = self.training_history.get('val_loss', [0])[-1]

            self.early_stopped = self.epochs_completed < self.epochs

            # Save final model
            final_model_path = os.path.join(self.output_dir, "final_model")
            self._save_model(final_model_path)
            self.model_saved_path = final_model_path

            # Record model artifacts
            model_hash = self.provenance_tracker.record_artifact(
                final_model_path,
                "model",
                "final_model",
                {
                    "model_type": self.model.__class__.__name__,
                    "training_time": self.training_duration,
                    "epochs_completed": self.epochs_completed,
                    "best_val_loss": self.best_val_loss,
                    "final_train_loss": self.final_train_loss,
                    "final_val_loss": self.final_val_loss
                }
            )

            if self.best_model_saved_path:
                best_model_hash = self.provenance_tracker.record_artifact(
                    self.best_model_saved_path,
                    "model",
                    "best_model",
                    {
                        "model_type": self.model.__class__.__name__,
                        "best_val_loss": self.best_val_loss,
                        "epoch": self.best_epoch
                    }
                )

            # Save training history
            history_path = os.path.join(self.output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_serializable = {}
                for key, values in self.training_history.items():
                    if isinstance(values, np.ndarray):
                        history_serializable[key] = values.tolist()
                    elif isinstance(values, list):
                        history_serializable[key] = values
                    else:
                        history_serializable[key] = [values]

                json.dump(history_serializable, f, indent=2)

            # Record training history artifact
            self.provenance_tracker.record_artifact(
                history_path,
                "metadata",
                "training_history",
                {
                    "total_epochs": self.epochs_completed,
                    "metrics": list(self.training_history.keys()),
                    "early_stopped": self.early_stopped
                }
            )

            # Store training parameters in database
            self.provenance_tracker.db_api.store_parameters(
                self.provenance_tracker.run_id, {
                    "epochs_requested": self.epochs,
                    "epochs_completed": self.epochs_completed,
                    "batch_size": self.batch_size,
                    "training_duration": self.training_duration,
                    "early_stopped": self.early_stopped,
                    "best_val_loss": self.best_val_loss,
                    "best_epoch": self.best_epoch,
                    "final_train_loss": self.final_train_loss,
                    "final_val_loss": self.final_val_loss
                }, "training"
            )

            # Record final training metrics
            final_metrics = {
                "training_time_seconds": self.training_duration,
                "epochs_completed": self.epochs_completed,
                "best_val_loss": self.best_val_loss,
                "final_train_loss": self.final_train_loss,
                "final_val_loss": self.final_val_loss,
                "early_stopped": self.early_stopped
            }

            self.provenance_tracker.record_metrics(final_metrics)

        # Return training results
        results = {
            "epochs_completed": self.epochs_completed,
            "best_val_loss": self.best_val_loss,
            "training_time": self.training_duration,
            "best_model_path": self.best_model_saved_path,
            "final_model_path": final_model_path,
            "training_history": self.training_history,
            "model_hash": model_hash if 'model_hash' in locals() else None
        }

        logger.info(
            f"Model training completed in {self.training_duration:.2f} seconds with best val_loss: {self.best_val_loss:.4f}")
        return results

    def evaluate(self, model=None) -> Dict[str, Any]:
        """Evaluate a model on the test dataset with artifact tracking."""
        logger.info("Evaluating model on test data")

        if model is None:
            model = self.model

        # Get test data
        X_test, y_test = self.data_manager.get_test_data()

        evaluation_params = {
            "test_samples": len(X_test),
            "model_type": model.__class__.__name__
        }

        with self.provenance_tracker.track_stage("evaluation", evaluation_params):
            # Record start time
            start_time = time.time()

            # Make predictions
            y_pred = model.predict(X_test)

            # Initialize metrics dictionary
            metrics = {}
            test_loss = 0.0

            # Core evaluation - prioritize the model's own evaluation methods
            if hasattr(model, "evaluate"):
                # Get basic metrics from model's evaluate method
                eval_result = model.evaluate(X_test, y_test, verbose=0)

                # Handle different return types from evaluate
                if isinstance(eval_result, dict):
                    metrics.update(eval_result)
                    test_loss = eval_result.get("test_loss", eval_result.get("loss", 0.0))
                elif isinstance(eval_result, list) and hasattr(model, "metrics_names"):
                    # Handle Keras-style evaluate that returns a list
                    for i, metric_name in enumerate(model.metrics_names):
                        if i < len(eval_result):
                            metrics[f"test_{metric_name}"] = eval_result[i]

                    # First item is usually loss
                    test_loss = eval_result[0] if len(eval_result) > 0 else 0.0

                elif isinstance(eval_result, (list, tuple)):
                    # Handle case where we don't have metrics_names but have a list
                    test_loss = eval_result[0] if len(eval_result) > 0 else 0.0
                    metrics["test_loss"] = test_loss

                    # Add other metrics with generic names
                    for i, value in enumerate(eval_result[1:], 1):
                        metrics[f"test_metric_{i}"] = value

                elif isinstance(eval_result, (int, float)):
                    # Single scalar value
                    test_loss = float(eval_result)
                    metrics["test_loss"] = test_loss
                else:
                    logger.warning(f"Unexpected evaluation result type: {type(eval_result)}")
                    test_loss = 0.0
                    metrics["test_loss"] = test_loss

            # Get custom metrics directly from the model if available
            if hasattr(model, "custom_metrics"):
                try:
                    custom_metrics = model.custom_metrics(y_test, y_pred)
                    metrics.update(custom_metrics)

                    # Use model's loss value function if available
                    if hasattr(model, "get_loss_value") and "test_loss" not in metrics:
                        test_loss = model.get_loss_value(metrics)
                        metrics["test_loss"] = test_loss
                except Exception as e:
                    logger.warning(f"Error calculating custom metrics: {e}")

            # Ensure test_loss is a scalar
            if isinstance(test_loss, (list, tuple, np.ndarray)):
                test_loss = float(test_loss[0]) if len(test_loss) > 0 else 0.0
            else:
                test_loss = float(test_loss)

            # Record end time
            end_time = time.time()
            eval_time = end_time - start_time

            # Add evaluation metadata
            metrics.update({
                "evaluation_time_seconds": eval_time,
                "test_samples": len(X_test)
            })

            # Record evaluation metrics
            self.provenance_tracker.record_metrics(metrics)

            # Save evaluation results
            eval_results_path = os.path.join(self.output_dir, "evaluation_results.json")

            # Prepare results for serialization
            results_to_save = {
                "metrics": {k: v for k, v in metrics.items() if k not in ["y_pred", "y_true"]},
                "test_loss": test_loss,
                "evaluation_time": eval_time,
                "test_samples": len(X_test)
            }

            with open(eval_results_path, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)

            # Record evaluation results artifact
            self.provenance_tracker.record_artifact(
                eval_results_path,
                "metadata",
                "evaluation_results",
                {
                    "test_loss": test_loss,
                    "evaluation_time": eval_time,
                    "metrics_count": len(metrics)
                }
            )

            # Save predictions if configured
            if self.config.get("save_predictions", True):
                predictions_path = os.path.join(self.output_dir, "predictions.npz")
                np.savez_compressed(
                    predictions_path,
                    y_true=y_test,
                    y_pred=y_pred
                )

                self.provenance_tracker.record_artifact(
                    predictions_path,
                    "predictions",
                    "test_predictions",
                    {
                        "prediction_shape": y_pred.shape if hasattr(y_pred, 'shape') else len(y_pred),
                        "true_labels_shape": y_test.shape if hasattr(y_test, 'shape') else len(y_test)
                    }
                )

            # Add predictions and true values to results for report generation
            # Convert to regular Python lists for JSON serialization
            if isinstance(y_pred, np.ndarray):
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    # Multi-class probabilities - convert to class predictions
                    y_pred_for_results = np.argmax(y_pred, axis=1).tolist()
                else:
                    y_pred_for_results = y_pred.flatten().tolist()
            else:
                y_pred_for_results = y_pred

            if isinstance(y_test, np.ndarray):
                y_true_for_results = y_test.flatten().tolist()
            else:
                y_true_for_results = y_test

            metrics["y_pred"] = y_pred_for_results
            metrics["y_true"] = y_true_for_results

        logger.info(f"Model evaluation completed in {eval_time:.2f} seconds with test_loss: {test_loss:.4f}")
        return metrics

    def _setup_callbacks(self) -> List[Any]:
        """Set up training callbacks with artifact tracking."""
        callbacks = []

        # Early stopping callback
        if self.early_stopping.get("enabled", True):
            callbacks.append(self._create_early_stopping_callback())

        # Checkpoint callback with artifact tracking
        if self.checkpoint_config.get("enabled", True):
            callbacks.append(self._create_checkpoint_callback())

        # Real-time metrics tracking callback
        callbacks.append(self._create_training_metrics_callback())

        return callbacks

    def _create_early_stopping_callback(self) -> Any:
        """Create early stopping callback."""
        return EarlyStopping(
            monitor=self.early_stopping.get("monitor", "val_loss"),
            patience=self.early_stopping.get("patience", 10),
            restore_best_weights=True,
            verbose=1
        )

    def _create_checkpoint_callback(self) -> Any:
        """Create model checkpoint callback with artifact tracking."""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Simple checkpoint path to avoid format conflicts
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.keras")

        callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.checkpoint_config.get("monitor", "val_loss"),
            save_best_only=self.checkpoint_config.get("save_best_only", True),
            save_weights_only=False,
            verbose=1
        )

        # Store the path for later reference
        self.best_model_saved_path = checkpoint_path

        return callback

    def _create_training_metrics_callback(self):
        """Create callback for real-time training metrics tracking."""
        return TrainingMetricsCallback(self, self.provenance_tracker).callback

    def _save_model(self, path: str) -> str:
        """Save model to disk with proper artifact tracking."""
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

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
        """Get learning curves from training history."""
        if not self.training_history:
            logger.warning("No training history available")
            return {}

        return self.training_history

    def get_training_summary(self) -> Dict[str, Any]:
        """Get explicit training summary for reporting."""
        return {
            "training_duration": self.training_duration,
            "epochs_completed": self.epochs_completed,
            "epochs_requested": self.epochs,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "early_stopped": self.early_stopped,
            "model_saved_path": self.model_saved_path,
            "best_model_saved_path": self.best_model_saved_path,
            "batch_size": self.batch_size
        }