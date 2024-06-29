import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import Input, Model


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


def swish(x):
    return x * tf.sigmoid(x)


def create_model(params):
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
    model.add(Dense(layer_size, input_shape=(5,)))
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

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def run_model(x_train, y_train, x_val, y_val, params):
    batch_size = int(float(params["batch_size"]))
    epochs = 200
    model = create_model(params)

    # Define ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # Monitor validation loss
        factor=0.1,           # Factor by which the learning rate will be reduced (e.g., 0.5 means halving the LR)
        patience=5,           # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-7,          # Minimum learning rate
        verbose=1,             # 1: Update messages, 0: No update messages
    )

    # Define EarlyStopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=10,          # Number of epochs with no improvement after which training will be stopped
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
    print(f"best_epoch {np.argmin(history.history['val_loss'])+1}")
    return model, history
