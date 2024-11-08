import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ProgbarLogger
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import os
from tensorflow.keras.regularizers import l1, l2


def create_tof_to_energy_model(params, steps_per_execution):
    layer_size = int(params['layer_size'])
    dropout_rate = float(params['dropout'])
    learning_rate = float(params['learning_rate'])
    optimizer = params['optimizer']
    regularization = float(params.get('regularization', 0.01))  # Default regularization

    model = Sequential()

    # Input layer with L2 regularization
    model.add(Dense(layer_size, input_shape=(14,),
                    kernel_regularizer=l2(regularization)))
    model.add(LeakyReLU(alpha=0.02))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Hidden layers with L2 regularization
    model.add(Dense(layer_size * 2, kernel_regularizer=l2(regularization)))
    model.add(LeakyReLU(alpha=0.02))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(layer_size, kernel_regularizer=l2(regularization)))
    model.add(Activation('swish'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Optimizer setup
    if optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  steps_per_execution=steps_per_execution)

    return model


def train_tof_to_energy_model(train_gen, val_gen, params, checkpoint_dir):
    epochs = params.get('epochs', 200)
    steps_per_epoch = params['steps_per_epoch']
    validation_steps = params['validation_steps']
    steps_per_execution = steps_per_epoch // 10

    model = create_tof_to_energy_model(params, steps_per_execution)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    )
    checkpoint_path = os.path.join(checkpoint_dir, "main_cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', save_best_only=True,
        save_weights_only=True, verbose=1
    )
    # memory_callback = MemoryUsageCallback()

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
    print(f"best_epoch {np.argmin(history.history['val_loss']) + 1}")
    return model, history
