import os
import logging
import numpy as np
import tensorflow as tf
from typing import Optional, Dict
from abc import ABC
from src.tof_ml.models.model_factory import ModelFactory

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

logger = logging.getLogger(__name__)

class Trainer(ABC):
    def __init__(
        self,
        config: Optional[Dict] = None,
        model_config: Optional[Dict] = None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        output_path: str = "./artifacts",
        meta_data: Optional[Dict] = None,
        **kwargs
    ):
        """
        If 'config' is provided, we'll load fields from it (e.g., model_config).
        Then direct parameters override the config-based values.
        We store relevant training/eval info in 'meta_data'.
        """
        # 1) Default internal attributes
        self.config = {}
        self.model_config = {}
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        # Data splits
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.model = None
        self.best_val_mse = None
        self.test_mse = None

        # We'll store all relevant info here
        self.meta_data = {} if meta_data is None else meta_data

        # 2) If a config is provided, load from config
        if config is not None:
            self._init_from_config(config, **kwargs)

        # 3) Override with direct parameters if they're not None
        if model_config is not None:
            self.model_config = model_config

        if X_train is not None:
            self.X_train = X_train
        if y_train is not None:
            self.y_train = y_train
        if X_val is not None:
            self.X_val = X_val
        if y_val is not None:
            self.y_val = y_val
        if X_test is not None:
            self.X_test = X_test
        if y_test is not None:
            self.y_test = y_test

    def _init_from_config(self, config: Dict, **kwargs):
        """
        A helper to read any relevant fields from the config.
        e.g. config.get("model").
        """
        self.config = config
        self.model_config = config.get("model", {})

    def run_training(self):
        """
        Core training routine:
          - Build model from self.model_config
          - Train with Keras
          - Save final model
          - Store best val MSE in meta_data
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data provided to Trainer. Cannot run training.")

        # 1) Build the model (assuming you have a ModelFactory utility)
        self.model = ModelFactory.create_model(self.model_config)
        logger.info("Model created from model_config.")

        # 2) Retrieve hyperparams from self.model_config["params"]
        params = self.model_config.get("params", {})
        epochs = params.get("epochs", 100)
        batch_size = params.get("batch_size", 32)
        monitor_metric = params.get("monitor_metric", "val_loss")

        # EarlyStopping
        early_stopping_patience = params.get("early_stopping_patience", 10)
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            restore_best_weights=True
        )

        # ReduceLROnPlateau
        reduce_lr_factor = params.get("reduce_lr_factor", 0.2)
        reduce_lr_patience = params.get("reduce_lr_patience", 20)
        reduce_lr_min_lr = params.get("reduce_lr_min_lr", 1e-6)
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=reduce_lr_min_lr,
            verbose=1
        )

        # ModelCheckpoint
        best_model_path = os.path.join(self.output_path, "best_model.h5")
        ckpt = ModelCheckpoint(
            filepath=best_model_path,
            monitor=monitor_metric,
            save_best_only=True,
            verbose=1
        )

        logger.info(f"Starting training with batch_size={batch_size}, epochs={epochs}...")

        # 3) Convert arrays to tf.data.Dataset
        train_ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        val_ds = val_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # 4) Fit the model
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            callbacks=[ckpt, early_stop, reduce_lr],
        )

        logger.info("Training complete.")

        # 5) Best val MSE from history
        if "val_loss" in history.history:
            self.best_val_mse = float(min(history.history["val_loss"]))
            logger.info(f"Best epoch val MSE: {self.best_val_mse:.4f}")
        else:
            self.best_val_mse = None

        # 6) Save final model
        final_model_path = os.path.join(self.output_path, "final_model.h5")
        self.model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        # 7) Update trainer metadata
        self._update_training_metadata(history, final_model_path, params)

    def evaluate_model(self):
        """
        Evaluate on test data, compute MSE, store it in self.test_mse.
        Now also return predictions, true y, and residuals so we
        can produce the new plots without calling evaluate again.
        """
        if self.model is None:
            raise ValueError("No model found. Did you run run_training()?")

        if self.X_test is None or self.y_test is None:
            logger.warning("No test data provided to Trainer.")
            return None, None, None

        y_test_pred = self.model.predict(self.X_test).flatten()
        y_test_true = self.y_test.flatten()
        residuals = y_test_true - y_test_pred

        self.test_mse = float(np.mean(residuals ** 2))
        logger.info(f"Test MSE: {self.test_mse:.4f}")

        # Store predictions and residuals for reference if needed
        self.y_test_pred = y_test_pred
        self.residuals = residuals

        # Update metadata
        self._update_evaluation_metadata()

        return y_test_pred, y_test_true, residuals

    def _update_training_metadata(self, history, final_model_path: str, params: Dict):
        """
        Store training info in meta_data so we can review or log it elsewhere.
        """
        self.meta_data["trainer_class"] = self.__class__.__name__
        self.meta_data["final_model_path"] = final_model_path
        self.meta_data["best_val_mse"] = self.best_val_mse

        # Store all relevant hyperparameters
        self.meta_data["training_params"] = dict(params)

        # Optionally store full training history
        if hasattr(history, "history"):
            self.meta_data["training_history"] = {
                k: [float(val) for val in v] for k, v in history.history.items()
            }

    def _update_evaluation_metadata(self):
        """
        Store test evaluation info in meta_data.
        """
        self.meta_data["test_mse"] = self.test_mse

