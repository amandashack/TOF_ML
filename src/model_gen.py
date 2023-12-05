import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Activation

# make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)


# Define Swish activation function
def swish(x):
    return x * tf.sigmoid(x)


# Defining the Y0 from the regression line on the NM simulation datasets
def y0_NM(x_value):
    y0 = -0.4986 * x_value - 0.5605
    return y0


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
    dropout = params['dropout']
    layer_size = int(params['layer_size'])
    alpha = params['alpha']

    model = Sequential()
    model.add(Dense(layer_size, input_shape=(6,)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(layer_size*2))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(layer_size, activation=swish))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model


def run_model(x_train, y_train, x_val, y_val, params):
    batch_size = int(params["batch_size"])
    epochs = int(params["epochs"])
    model = create_model(params)
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, validation_data=(x_val, y_val), verbose=0)
    return model, history
