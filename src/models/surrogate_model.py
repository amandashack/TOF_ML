import tensorflow as tf
import os

def swish(x):
    return x * tf.sigmoid(x)

def create_main_model(params, steps_per_execution):
    layer_size = int(params['layer_size'])
    dropout_rate = float(params['dropout'])
    learning_rate = float(params['learning_rate'])
    optimizer = params['optimizer']

    inputs = tf.keras.Input(shape=(18,))
    x = tf.keras.layers.Dense(layer_size)(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(layer_size / 2)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    time_of_flight_output = tf.keras.layers.Dense(1, activation='linear', name='time_of_flight')(x)
    y_tof_output = tf.keras.layers.Dense(1, activation='linear', name='y_tof')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[time_of_flight_output, y_tof_output])

    if optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss={'time_of_flight': 'mean_squared_error', 'y_tof': 'mean_squared_error'},
        optimizer=optimizer,
        steps_per_execution=steps_per_execution
    )

    return model


class MemoryUsageCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        #print(f" - Batch {batch + 1} - GPU Memory Usage: {memory_info['current'] / 1024**2:.2f} "
        #      f"MB / {memory_info['peak'] / 1024**2:.2f} MB")


def train_main_model(train_gen, val_gen, params, checkpoint_dir):
    epochs = params.get('epochs', 200)
    steps_per_epoch = params['steps_per_epoch']
    validation_steps = params['validation_steps']
    steps_per_execution = steps_per_epoch // 10

    model = create_main_model(params, steps_per_execution)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5, verbose=0
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    checkpoint_path = os.path.join(checkpoint_dir, "main_cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', save_best_only=True,
        save_weights_only=True, verbose=0
    )
    #memory_callback = MemoryUsageCallback()

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[reduce_lr, early_stop, checkpoint]
    )

    return model, history
