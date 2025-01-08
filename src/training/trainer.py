# src/training/trainer.py

import os
import logging
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.models.model_factory import ModelFactory
from src.utils.plotting_tools import evaluate_and_plot_test


class Trainer:
    def __init__(self,
                 model_config: dict,
                 training_config: dict,
                 logger: logging.Logger,
                 db_api,
                 df,
                 output_path: str = "./artifacts"):
        """
        :param model_config: Model hyperparameters & type from base_config
        :param training_config: Training-related config (split sizes, scaling, etc.)
        :param logger: Logger for info/warnings
        :param db_api: Database API for recording runs
        :param df: The filtered DataFrame from your pipeline
        :param output_path: Where to save models/plots
        """
        self.model_config = model_config
        self.training_config = training_config
        self.logger = logger
        self.db_api = db_api
        self.df = df
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        # These will be created in prepare_data()
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.scaler = None
        self.model = None
        self.best_val_mse = None
        self.test_mse = None

    def prepare_data(self):
        """
        1) Splits df into train/val/test
        2) Scales if specified
        3) Adds interaction terms to X_train/X_val/X_test
           or does any custom transformations.
        """
        # 1) Extract arrays from df
        feature_cols = self.training_config["features"]["input_columns"]
        output_col = self.training_config["features"]["output_column"]

        self.logger.info("Select X and y.")

        X = self.df[:, [7, 6]]
        y = self.df[:, 0]

        # 2) Split into train+val vs test
        test_ratio = self.training_config.get("test_ratio", 0.1)
        random_state = self.training_config.get("random_state", 42)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state
        )

        # 3) Now split X_temp -> train vs val
        val_ratio = self.training_config.get("val_ratio", 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )

        # 4) Scale if needed
        scaler_type = self.training_config.get("scaler", "None")
        if scaler_type == "StandardScaler":
            self.scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        # 5) Add interaction terms (or any custom transformations)
        #    For example, if you want to do log2(X_train[:,0]) * X_train[:,1], etc.
        #    We'll demonstrate a simple function call for each subset.
        if self.training_config["features"].get("generate_interactions", False):
            X_train = self._add_interactions(X_train)
            X_val = self._add_interactions(X_val)
            X_test = self._add_interactions(X_test)

        # Store them in class attributes
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val,   y_val
        self.X_test, self.y_test = X_test,  y_test

        self.logger.info(f"Data prepared. Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    def run_training(self):
        """
        Builds and trains the model using X_train/X_val from prepare_data().
        Tracks best epoch MSE. Saves final model.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Please call prepare_data() before run_training().")

        # Build the model
        self.model = ModelFactory.create_model(self.model_config)

        # Keras callbacks
        best_model_path = os.path.join(self.output_path, "best_model.h5")
        ckpt = ModelCheckpoint(filepath=best_model_path, monitor="val_loss",
                               save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3,
                                      min_lr=1e-6, verbose=1)

        # Fit the model
        self.logger.info("Starting training with Keras model...")
        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            callbacks=[ckpt, early_stop, reduce_lr]
        )

        self.logger.info("Training complete.")

        # Best val MSE is min of val_loss
        self.best_val_mse = float(min(history.history["val_loss"]))
        self.logger.info(f"Best epoch val MSE: {self.best_val_mse:.4f}")

        # Save final model (after best weights restored)
        final_model_path = os.path.join(self.output_path, "final_model.h5")
        self.model.save(final_model_path)
        self.logger.info(f"Final model saved to: {final_model_path}")

    def evaluate_model(self):
        """
        Evaluate on test set. Generate test plots.
        """
        if self.model is None:
            raise ValueError("No model found. Did you run run_training()?")

        # Basic MSE
        y_test_pred = self.model.predict(self.X_test)
        self.test_mse = float(np.mean((self.y_test - y_test_pred) ** 2))
        self.logger.info(f"Test MSE: {self.test_mse:.4f}")

        # If you want to also generate advanced plots:
        test_plots_dir = os.path.join(self.output_path, "test_plots")
        # e.g. evaluate_and_plot_test is your function from plotting_tools
        test_mse, plot_paths = evaluate_and_plot_test(
            model=self.model,
            X_test=self.X_test,
            y_test=self.y_test,
            scaler_y=None,  # or self.y_scaler if you used one
            output_dir=test_plots_dir,
            prefix="test"
        )
        # Overwrite test_mse if you prefer the function's computed MSE
        self.test_mse = test_mse
        self.logger.info(f"Test MSE (via evaluate_and_plot_test): {self.test_mse:.4f}")

        # Optionally store plot_paths if you want to upload them in record_to_database
        self.test_plot_paths = plot_paths

    def record_to_database(self, base_config):
        """
        Uses db_api to record final results. We'll store:
          - final_train_mse or best_val_mse
          - test_mse
          - entire base_config
          - model path, etc.
        """
        if self.test_mse is None:
            self.logger.warning("Test MSE is None. Did you call evaluate_model()?")

        # Construct training_results dict with relevant metrics
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
        self.logger.info("Recorded results to Notion DB.")

    def _add_interactions(self, X: np.ndarray) -> np.ndarray:
        """
        Example: X has 2 columns => [x1, x2]
        We'll log2(x1), keep x2 as-is, then add x1*x2, x1^2, x2^2, etc.
        Return the new array with shape (N, 5).
        Adjust logic as needed for your data shape.
        """
        if X.shape[1] < 2:
            self.logger.warning("X does not have at least 2 columns, skipping interaction terms.")
            return X

        x1 = X[:, 0]
        x2 = X[:, 1]

        # log2(x1), keep x2
        x1_log = np.log2(np.clip(x1, 1e-9, None))  # avoid log(0)
        interactions = np.column_stack([
            x1_log * x2,
            x1_log**2,
            x2**2
        ])
        new_X = np.column_stack([x1_log, x2, interactions])
        return new_X

