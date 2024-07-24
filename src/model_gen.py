import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ProgbarLogger
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import Input, Model
import os
import matplotlib.pyplot as plt


# Define Swish activation function
def swish(x):
    return x * tf.sigmoid(x)


def create_model(params, steps_per_execution):
    """
        Model providing function:

        Create Keras model with double curly brackets dropped-in as needed.
        Return value has to be a valid python dictionary with two customary keys:
            - loss: Specify a numeric evaluation metric to be minimized
            - status: Just use STATUS_OK and see hyperopt documentation if not feasible
        The last one is optional, though recommended, namely:
            - model: specify the model just created so that we can later use it again.
        """
    layer_size = int(params['layer_size'])
    dropout_rate = float(params['dropout'])
    learning_rate = float(params['learning_rate'])
    optimizer = params['optimizer']

    model = Sequential()

    # Input layer
    model.add(Dense(layer_size, input_shape=(18,)))
    model.add(LeakyReLU(alpha=0.02))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Hidden layers
    model.add(Dense(layer_size * 2))
    model.add(LeakyReLU(alpha=0.02))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(layer_size))
    model.add(Activation('swish'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='linear'))

    if optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        steps_per_execution=steps_per_execution
    )

    return model


class MemoryUsageCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f" - Batch {batch + 1} - GPU Memory Usage: {memory_info['current'] / 1024**2:.2f} "
              f"MB / {memory_info['peak'] / 1024**2:.2f} MB")


# Define the learning rate finder class
class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr=1e-8, max_lr=10, steps_per_epoch=None, epochs=1):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_steps = steps_per_epoch * epochs
        self.step = 0
        self.lr_mult = (max_lr / min_lr) ** (1 / self.total_steps)
        self.losses = []
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.step += 1
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        lr *= self.lr_mult
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrs.append(lr)
        self.losses.append(np.log(logs['loss']))

        # Debugging: Print the learning rate and loss
        print(f"Step: {self.step}, Learning Rate: {lr}, Loss: {logs['loss']}")

        if self.step >= self.total_steps:
            self.model.stop_training = True

    def plot_loss(self, sma=1):
        # Apply simple moving average (SMA) to smooth the loss plot
        def sma_smoothing(values, window):
            weights = np.repeat(1.0, window) / window
            return np.convolve(values, weights, 'valid')

        smoothed_losses = sma_smoothing(self.losses, sma)

        plt.plot(self.lrs[:len(smoothed_losses)], smoothed_losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.show()


def run_model(train_dataset, val_dataset, params, checkpoint_dir):
    epochs = params.get('epochs', 50)
    steps_per_epoch = params['steps_per_epoch']
    validation_steps = params['validation_steps']
    steps_per_execution = steps_per_epoch // 10

    model = create_model(params, steps_per_execution)

    # Define callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, min_lr=1e-8, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=6, restore_best_weights=True
    )
    checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1
    )
    memory_callback = MemoryUsageCallback()

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Train the model with callbacks
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[reduce_lr, early_stop, checkpoint, memory_callback]
    )

    return model, history
