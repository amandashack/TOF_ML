# src/models/mlp_keras_regressor.py

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


class MLPKerasRegressor:
    """
    A simple Keras-based MLP regressor with a scikit-learn-like interface.
    Takes hyperparams in __init__ and uses them in fit().
    """

    def __init__(
            self,
            hidden_layers=[32, 64, 32],
            activations=["leaky_relu", "leaky_relu", "swish"],
            learning_rate=1e-3,
            optimizer_name="Adam",
            epochs=50,
            batch_size=32,
            regularization=0.0,
            dropout=0.2,
            **kwargs
    ):
        """
        Store parameters. Build model in _build_model().
        """
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.dropout = dropout

        self.model = None
        self._build_model()

    def _build_model(self):
        """
        Build a Sequential model using self.hidden_layers, self.activations, etc.
        """
        model = Sequential()

        # Example: input_dim is unknown at init,
        # so we handle dynamic shape in fit(...) or specify input_shape in constructor.
        # For simplicity, we won't specify input shape here. Keras can infer on first .fit().
        for i, (units, act) in enumerate(zip(self.hidden_layers, self.activations)):
            # If first layer:
            if i == 0:
                # We'll just define input_dim in fit() or let Keras infer
                model.add(layers.Dense(units, kernel_regularizer=l2(self.regularization)))
            else:
                model.add(layers.Dense(units, kernel_regularizer=l2(self.regularization)))

            # Activation
            if act.lower() == "leaky_relu":
                model.add(layers.LeakyReLU(alpha=0.02))
            else:
                model.add(layers.Activation(act))

            # Optional: dropout
            if self.dropout > 0:
                model.add(layers.Dropout(self.dropout))

        # Output layer
        model.add(layers.Dense(1, activation="linear"))

        # Choose optimizer
        if self.optimizer_name == "Adam":
            opt = Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == "SGD":
            opt = SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name == "RMSprop":
            opt = RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer {self.optimizer_name}")

        model.compile(loss="mse", optimizer=opt)
        self.model = model

    def fit(self,
            X,
            y,
            validation_data=None,
            callbacks=None,
            verbose=1
        ):
        """
        Fit the model using *internal* epochs, batch_size.
        Accepts optional validation_data, callbacks, verbose.

        We do *not* allow the user to pass epochs=..., batch_size=...
        here to avoid TypeError. These are set in the constructor.
        """
        if callbacks is None:
            callbacks = []

        history = self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        return history

    def predict(self, X):
        """
        Return predictions as a 1D numpy array.
        """
        preds = self.model.predict(X).flatten()
        return preds

    def save(self, filepath):
        """
        Save the entire model (architecture + weights).
        """
        self.model.save(filepath)
