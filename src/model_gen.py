import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import Input, Model


# Define Swish activation function
def swish(x):
    return x * tf.sigmoid(x)


# Custom loss function to incorporate mask
def time_of_flight_loss(y_true, y_pred):
    time_of_flight_true, mask = y_true[:, 0], y_true[:, 2]
    mask = tf.cast(mask, dtype=tf.float32)
    loss = tf.square(time_of_flight_true - y_pred)
    return tf.reduce_mean(mask * loss)

def y_tof_loss(y_true, y_pred):
    y_tof_true, mask = y_true[:, 1], y_true[:, 2]
    mask = tf.cast(mask, dtype=tf.float32)
    loss = tf.square(y_tof_true - y_pred)
    return tf.reduce_mean(mask * loss)


def create_model(params):
    layer_size = int(params['layer_size'])
    dropout_rate = float(params['dropout'])
    learning_rate = float(params['learning_rate'])
    optimizer = params['optimizer']

    inputs = tf.keras.Input(shape=(5,))
    x = tf.keras.layers.Dense(layer_size)(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(layer_size * 2)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(layer_size)(x)
    x = tf.keras.layers.Activation(swish)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    time_of_flight_output = tf.keras.layers.Dense(1, activation='linear', name='time_of_flight')(x)
    y_tof_output = tf.keras.layers.Dense(1, activation='linear', name='y_tof')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[time_of_flight_output, y_tof_output])

    if optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        loss={'time_of_flight': time_of_flight_loss, 'y_tof': y_tof_loss},
        optimizer=optimizer
    )

    return model


def run_model(x_train, y_train, x_val, y_val, params):
    batch_size = int(float(params["batch_size"]))
    epochs = 200
    model = create_model(params)

    # Define ReduceLROnPlateau callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Monitor validation loss
        factor=0.1,  # Factor by which the learning rate will be reduced (e.g., 0.5 means halving the LR)
        patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-7,  # Minimum learning rate
        verbose=1  # 1: Update messages, 0: No update messages
    )

    # Define EarlyStopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restore model weights to the best observed during training
    )

    # Train the model with callbacks
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=1,
        callbacks=[reduce_lr, early_stop]  # Add callbacks here
    )
    print(f"early_stop {early_stop.stopped_epoch}")
    print(f"best_epoch {np.argmin(history.history['val_loss']) + 1}")
    return model, history
