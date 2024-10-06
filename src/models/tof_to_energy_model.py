import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, Activation, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ProgbarLogger
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import re
import os
import fcntl


class MetaFileCallback(tf.keras.callbacks.Callback):
    def __init__(self, param_ID, job_name, meta_file):
        super(MetaFileCallback, self).__init__()
        self.param_ID = param_ID
        self.job_name = job_name
        self.meta_file = meta_file

    def on_epoch_end(self, epoch, logs=None):
        # Update the meta file with the latest checkpoint number (epoch + 1)
        try:
            with open(self.meta_file, 'r+') as f:
                # Acquire exclusive lock
                fcntl.flock(f, fcntl.LOCK_EX)
                lines = f.readlines()
                f.seek(0)
                f.truncate()
                found = False
                for line in lines:
                    line = line.strip()
                    if line.startswith(f"{self.param_ID}|{self.job_name}|"):
                        # Update the line with the latest checkpoint number
                        new_line = f"{self.param_ID}|{self.job_name}|{epoch + 1}"
                        f.write(new_line + '\n')
                        found = True
                    else:
                        f.write(line + '\n')
                if not found:
                    # If the line is not found, add it
                    new_line = f"{self.param_ID}|{self.job_name}|{epoch + 1}"
                    f.write(new_line + '\n')
                # Release the lock
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Error updating meta file: {e}")


def create_tof_to_energy_model(params, steps_per_execution):
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
    job_name = params['job_name']

    model = Sequential()
    if job_name == 'tofs_simple_relu':
        # Architecture with two dense layers, activations relu and relu
        model.add(Dense(layer_size, activation='relu', input_shape=(11,)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(layer_size // 2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    elif job_name == 'tofs_simple_swish':
        # Architecture with two dense layers, activations relu and swish
        model.add(Dense(layer_size, activation='relu', input_shape=(11,)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(layer_size // 2, activation='swish'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    else:
        # Default architecture with three dense layers, activations relu, relu, swish
        model.add(Dense(layer_size, activation='relu', input_shape=(11,)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(layer_size * 2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(layer_size, activation='swish'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    # Output layer
    model.add(Dense(1, activation='linear'))

    # Learning rate scheduling
    if optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    if optimizer == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    if optimizer == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer,
                  steps_per_execution=steps_per_execution)

    return model


def train_tof_to_energy_model(train_gen, val_gen, params, checkpoint_dir, param_ID, job_name, meta_file):
    epochs = params.get('epochs', 200)
    steps_per_epoch = params['steps_per_epoch']
    validation_steps = params['validation_steps']
    steps_per_execution = steps_per_epoch // 10

    # Initialize the MirroredStrategy (we'll discuss this in the next section)
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Check for existing checkpoints
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Loading model from checkpoint: {latest_checkpoint}")
            model = tf.keras.models.load_model(latest_checkpoint)
        else:
            print("No checkpoint found. Initializing a new model.")
            model = create_tof_to_energy_model(params, steps_per_execution)

    # Learning rate scheduler and early stopping
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    # Checkpoint callback
    checkpoint_path = os.path.join(checkpoint_dir, "main_cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', save_best_only=False,
        save_weights_only=False, verbose=1
    )

    # Custom MetaFileCallback
    meta_callback = MetaFileCallback(param_ID, job_name, meta_file)

    # Callbacks list
    callbacks = [reduce_lr, early_stop, checkpoint]#, meta_callback]

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if latest_checkpoint:
        # Extract the epoch number from the checkpoint filename
        checkpoint_pattern = r"main_cp-(\d{4})\.ckpt"
        match = re.match(checkpoint_pattern, os.path.basename(latest_checkpoint))
        if match:
            initial_epoch = int(match.group(1))
            print(f"Resuming training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
    else:
        initial_epoch = 0

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        initial_epoch=initial_epoch
    )

    print(f"Early stopping at epoch {early_stop.stopped_epoch}")
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"Best epoch based on validation loss: {best_epoch}")
    return model, history