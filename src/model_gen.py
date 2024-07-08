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


# Custom loss function to incorporate mask
def time_of_flight_loss(y_true, y_pred):
    time_of_flight_true = y_true[:, 0]
    loss = tf.square(time_of_flight_true - y_pred)
    return tf.reduce_mean(loss)


def create_model(params, steps_per_execution):
    layer_size = int(params['layer_size'])
    dropout_rate = float(params['dropout'])
    learning_rate = float(params['learning_rate'])
    optimizer = params['optimizer']

    inputs = tf.keras.Input(shape=(14,), name="inputs")
    x = tf.keras.layers.Dense(layer_size, name="dense_1")(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.02, name="leakyrelu_1")(x)
    x = tf.keras.layers.BatchNormalization(name="batchnorm_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = tf.keras.layers.Dense(layer_size * 2, name="dense_2")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02, name="leakyrelu_2")(x)
    x = tf.keras.layers.BatchNormalization(name="batchnorm_2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_2")(x)

    x = tf.keras.layers.Dense(layer_size, name="dense_3")(x)
    x = tf.keras.layers.Activation(swish, name="swish")(x)
    x = tf.keras.layers.BatchNormalization(name="batchnorm_3")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_3")(x)

    time_of_flight_output = tf.keras.layers.Dense(1, activation='linear',
                                                  name='time_of_flight')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[time_of_flight_output])

    if optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        loss={'time_of_flight': time_of_flight_loss},
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
    epochs = 5
    steps_per_epoch = params['steps_per_epoch']
    validation_steps = params['validation_steps']
    steps_per_execution = steps_per_epoch // 10

    model = create_model(params, steps_per_execution)

    # Define ReduceLROnPlateau callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Monitor validation loss
        factor=0.1,  # Factor by which the learning rate will be reduced (e.g., 0.5 means halving the LR)
        patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-8,  # Minimum learning rate
        verbose=1  # 1: Update messages, 0: No update messages
    )

    # Define EarlyStopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restore model weights to the best observed during training
    )

    # Define ModelCheckpoint callback
    checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',  # Monitor validation loss
        save_best_only=True,  # Save only the best model
        save_weights_only=True,  # Save only the model weights
        verbose=1  # Verbosity mode
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
        steps_per_epoch=steps_per_epoch,  # Specify steps per epoch
        validation_steps=validation_steps,  # Specify validation steps
        verbose=1,  # Suppress the default verbose output
        callbacks=[reduce_lr, early_stop, checkpoint, memory_callback]  # Add callbacks here
    )

    print(f"early_stop {early_stop.stopped_epoch}")
    print(f"best_epoch {np.argmin(history.history['val_loss']) + 1}")
    return model, history
