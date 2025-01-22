# src/tof_ml/training/trainer.py

import os
import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.tof_ml.models.model_factory import ModelFactory
from src.tof_ml.utils.plotting_tools import evaluate_and_plot_test
from src.tof_ml.data.data_generator import DataGenerator

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,
                 model_config: dict,
                 training_config: dict,
                 db_api,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 output_path: str = "./artifacts"):
        self.model_config = model_config
        self.training_config = training_config
        self.db_api = db_api
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        # Store the splits
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.X_test  = X_test
        self.y_test  = y_test

        self.model = None
        self.best_val_mse = None
        self.test_mse = None
        self.test_plot_paths = {}

    def run_training(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data provided to Trainer.")

        # Build the model
        self.model = ModelFactory.create_model(self.model_config)

        # Keras callbacks
        best_model_path = os.path.join(self.output_path, "best_model.h5")
        ckpt = ModelCheckpoint(filepath=best_model_path, monitor="val_loss",
                               save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=20,
                                      min_lr=1e-6, verbose=1)

        batch_size = self.model_config.get("batch_size", 32)  # or read from training_config
        """train_gen = DataGenerator(self.X_train, self.y_train,
                                    batch_size=batch_size, shuffle=True)
        val_gen = DataGenerator(self.X_val, self.y_val,
                                  batch_size=batch_size, shuffle=False)
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, output_signature=(
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # mask as the target
            )
        ).take(len(self.X_train)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, output_signature=(
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # mask as the target
            )
        ).take(len(self.X_val)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)"""
        train_gen = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_gen = train_gen.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_gen = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        val_gen = val_gen.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # Fit
        logger.info("Starting training with Keras model...")
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            callbacks=[ckpt, early_stop, reduce_lr]
        )

        logger.info("Training complete.")

        # Best val MSE
        self.best_val_mse = float(min(history.history["val_loss"]))
        logger.info(f"Best epoch val MSE: {self.best_val_mse:.4f}")

        # Save final model
        final_model_path = os.path.join(self.output_path, "final_model.h5")
        self.model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("No model found. Did you run run_training()?")

        # Basic MSE in the final domain
        y_test_pred = self.model.predict(self.X_test).flatten()
        self.test_mse = float(np.mean((self.y_test - y_test_pred) ** 2))
        logger.info(f"Test MSE: {self.test_mse:.4f}")

        # Evaluate and plot
        test_plots_dir = os.path.join(self.output_path, "test_plots")
        test_mse, plot_paths = evaluate_and_plot_test(
            model=self.model,
            X_test=self.X_test,
            y_test=self.y_test,
            scaler_y=None,
            output_dir=test_plots_dir,
            prefix="test"
        )
        self.test_mse = test_mse
        logger.info(f"Test MSE (via evaluate_and_plot_test): {self.test_mse:.4f}")

        self.test_plot_paths = plot_paths

    def record_to_database(self, base_config):
        if self.test_mse is None:
            logger.warning("Test MSE is None. Did you call evaluate_model()?")

        training_results = {
            "val_mse":  self.best_val_mse,
            "test_mse": self.test_mse
        }

        model_path = os.path.join(self.output_path, "final_model.h5")
        self.db_api.record_model_run(
            config_dict=base_config,
            training_results=training_results,
            model_path=model_path,
            plot_paths=self.test_plot_paths
        )
        logger.info("Recorded results to Notion DB.")


