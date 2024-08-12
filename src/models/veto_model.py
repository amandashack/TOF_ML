import tensorflow as tf
import os


def create_veto_model(params):
    layer_size = int(params['layer_size'])
    dropout_rate = float(params['dropout'])
    learning_rate = float(params['learning_rate'])

    inputs = tf.keras.Input(shape=(18,), name="inputs")
    x = tf.keras.layers.Dense(layer_size, name="dense_1")(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.01, name="leakyrelu_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(x)

    hit_output = tf.keras.layers.Dense(1, activation='sigmoid', name='hit')(x)

    model = tf.keras.Model(inputs=inputs, outputs=hit_output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

def train_veto_model(train_dataset, val_dataset, params, checkpoint_dir):
    epochs = params.get('epochs', 200)
    steps_per_epoch = params['steps_per_epoch']
    validation_steps = params['validation_steps']

    model = create_veto_model(params)

    checkpoint_path = os.path.join(checkpoint_dir, "veto_cp-{epoch:04d}.ckpt")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, min_lr=1e-4, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=6, restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[reduce_lr, early_stop, checkpoint]
    )

    return model, history
