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


def time_of_flight_loss(y_true, y_pred):
    time_of_flight_true, mask_pred = y_true[:, 0], y_true[:, 2]
    mask = tf.cast(mask_pred, dtype=tf.float32)
    loss = tf.square(time_of_flight_true - y_pred)
    return tf.reduce_mean(mask * loss)


def y_tof_loss(y_true, y_pred):
    y_tof_true, mask_pred = y_true[:, 1], y_true[:, 2]
    mask = tf.cast(mask_pred, dtype=tf.float32)
    loss = tf.square(y_tof_true - y_pred)
    return tf.reduce_mean(mask * loss)


class CompositeModel(tf.keras.Model):
    def __init__(self, veto_model, main_model, **kwargs):
        super().__init__(**kwargs)
        self.veto_model = tf.keras.models.clone_model(veto_model)
        self.main_model = tf.keras.models.clone_model(main_model)

    def __call__(self, inputs):
        y_proba_hit = self.veto_model(inputs)
        y_hit = tf.squeeze(y_proba_hit > 0.5)
        hit_inputs = tf.boolean_mask(inputs, y_hit)
        tof_outputs, y_tof_outputs = self.main_model(hit_inputs)
        return y_hit, y_proba_hit, tof_outputs, y_tof_outputs


def create_veto_model(params):
    layer_size = int(params['layer_size'])
    dropout_rate = float(params['dropout'])
    learning_rate = float(params['learning_rate'])
    optimizer = params['optimizer']

    inputs = tf.keras.Input(shape=(18,), name="inputs")
    x = tf.keras.layers.Dense(layer_size, name="dense_1")(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.02, name="leakyrelu_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = tf.keras.layers.Dense(layer_size // 2, name="dense_2")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.02, name="leakyrelu_2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_2")(x)

    hit_output = tf.keras.layers.Dense(1, activation='sigmoid', name='hit')(x)

    model = tf.keras.Model(inputs=inputs, outputs=hit_output)

    if optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


def create_main_model(params, steps_per_execution):
    layer_size = int(params['layer_size'])
    dropout_rate = float(params['dropout'])
    learning_rate = float(params['learning_rate'])
    optimizer = params['optimizer']

    inputs = tf.keras.Input(shape=(18,), name="inputs")
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

    time_of_flight_output = tf.keras.layers.Dense(1, activation='linear', name='time_of_flight')(x)
    y_tof_output = tf.keras.layers.Dense(1, activation='linear', name='y_tof')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[time_of_flight_output, y_tof_output])

    if optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        loss={'time_of_flight': time_of_flight_loss, 'y_tof': y_tof_loss},
        optimizer=optimizer,
        steps_per_execution=steps_per_execution
    )

    return model


class MemoryUsageCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f" - Batch {batch + 1} - GPU Memory Usage: {memory_info['current'] / 1024**2:.2f} "
              f"MB / {memory_info['peak'] / 1024**2:.2f} MB")


def train_veto_model(train_dataset, val_dataset, params, checkpoint_dir):
    epochs = params.get('epochs', 50)
    steps_per_epoch = params['steps_per_epoch']
    validation_steps = params['validation_steps']

    model = create_veto_model(params)

    # Load the latest checkpoint if it exists
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir, model_name="veto_cp")
    if latest_checkpoint:
        print(f"Loading checkpoint: {latest_checkpoint}")
        model.load_weights(latest_checkpoint)

    checkpoint_path = os.path.join(checkpoint_dir, "veto_cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint]
    )

    return model, history


def train_main_model(train_gen, val_gen, params, checkpoint_dir):
    epochs = params.get('epochs', 50)
    steps_per_epoch = params['steps_per_epoch']
    validation_steps = params['validation_steps']
    steps_per_execution = steps_per_epoch // 10

    model = create_main_model(params, steps_per_execution)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, min_lr=1e-8, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=6, restore_best_weights=True
    )
    checkpoint_path = os.path.join(checkpoint_dir, "main_cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', save_best_only=True,
        save_weights_only=True, verbose=1
    )
    memory_callback = MemoryUsageCallback()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[reduce_lr, early_stop, checkpoint, memory_callback]
    )

    return model, history
