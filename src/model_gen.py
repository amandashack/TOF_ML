import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ProgbarLogger
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import Input, Model
import os
import matplotlib.pyplot as plt
import glob
import re


def get_latest_checkpoint(checkpoint_dir, model_name="veto_cp"):
    checkpoint_pattern = os.path.join(checkpoint_dir, f"{model_name}-*.index")
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    latest_checkpoint = latest_checkpoint.replace('.index', '')
    return latest_checkpoint


def get_best_checkpoint(checkpoint_dir, model_name="veto_cp"):
    checkpoint_pattern = os.path.join(checkpoint_dir, f"{model_name}-*.index")
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None
    best_checkpoint = None
    best_val_loss = float('inf')
    for checkpoint in checkpoints:
        # Extract the epoch and validation loss from the checkpoint file name
        match = re.search(rf"{model_name}-(\d+).index", checkpoint)
        if match:
            epoch = int(match.group(1))
            val_loss_file = checkpoint.replace(".index", ".val_loss")
            if os.path.exists(val_loss_file):
                with open(val_loss_file, 'r') as f:
                    val_loss = float(f.read().strip())
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = checkpoint.replace('.index', '')
    return best_checkpoint


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
    model.add(Dense(layer_size, input_shape=(14,)))
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

    # Learning rate scheduling
    if optimizer=="Adam":
        optimizer = Adam(learning_rate=learning_rate)
    if optimizer=="SGD":
        optimizer = SGD(learning_rate=learning_rate)
    if optimizer=="RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  steps_per_execution=steps_per_execution)

    return model


class MemoryUsageCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f" - Batch {batch + 1} - GPU Memory Usage: {memory_info['current'] / 1024**2:.2f} "
              f"MB / {memory_info['peak'] / 1024**2:.2f} MB")


def train_main_model(train_gen, val_gen, params, checkpoint_dir):
    epochs = params.get('epochs', 50)
    steps_per_epoch = params['steps_per_epoch']
    validation_steps = params['validation_steps']
    steps_per_execution = steps_per_epoch // 10

    model = create_model(params, steps_per_execution)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, min_lr=1e-4, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    checkpoint_path = os.path.join(checkpoint_dir, "main_cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', save_best_only=True,
        save_weights_only=True, verbose=1
    )
    #memory_callback = MemoryUsageCallback()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[reduce_lr, early_stop, checkpoint]
    )
    
    print(f"early_stop {early_stop.stopped_epoch}")
    print(f"best_epoch {np.argmin(history.history['val_loss'])+1}")
    return model, history
