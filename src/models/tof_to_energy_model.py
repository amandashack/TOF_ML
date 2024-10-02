import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ProgbarLogger
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import os


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
    if optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    if optimizer == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    if optimizer == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer,
                  steps_per_execution=steps_per_execution)

    return model


def train_tof_to_energy_model(train_gen, val_gen, params, checkpoint_dir):
    epochs = params.get('epochs', 200)
    steps_per_epoch = params['steps_per_epoch']
    validation_steps = params['validation_steps']
    steps_per_execution = steps_per_epoch // 10

    # Check for existing checkpoints
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Loading model from checkpoint: {latest_checkpoint}")
        model = create_tof_to_energy_model(params, steps_per_execution)
        model.load_weights(latest_checkpoint)
    else:
        print("No checkpoint found. Initializing a new model.")
        model = create_tof_to_energy_model(params, steps_per_execution)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    checkpoint_path = os.path.join(checkpoint_dir, "main_cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', save_best_only=True,
        save_weights_only=True, verbose=1
    )

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    initial_epoch = 0
    if latest_checkpoint:
        # Extract the epoch number from the checkpoint filename
        initial_epoch = int(latest_checkpoint.split('-')[-1].split('.')[0])
        print(f"Resuming training from epoch {initial_epoch}")

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[reduce_lr, early_stop, checkpoint],
        initial_epoch=initial_epoch
    )

    print(f"early_stop {early_stop.stopped_epoch}")
    print(f"best_epoch {np.argmin(history.history['val_loss']) + 1}")
    return model, history