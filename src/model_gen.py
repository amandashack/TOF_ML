import numpy as np
import os
import re
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Activation
import tqdm

# make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)
BUFFER_SIZE = 1500
BATCH_SIZE = 80
NUM_EPOCHS = 5


# Define Swish activation function
def swish(x):
    return x * tf.sigmoid(x)


# Defining the Y0 from the regression line on the NM simulation datasets
def y0_NM(x_value):
    y0 = -0.4821 * x_value - 0.7139
    return y0


def create_model1(X_train, Y_train, X_test, Y_test):
    epochs_list = [5] * 20

    # Create a PDF to store the plots
    pdf_filename = 'tof_prediction_plots_residual.pdf'
    pdf_pages = PdfPages(pdf_filename)

    model = Sequential()

    model.add(Dense(32, input_dim=3))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(16, activation=swish))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    loss_list = []

    # Create a loop to train and plot the model for each number of epochs
    for epochs in epochs_list:
        print(f"Training the model with {epochs} epochs; Residual Method")

        # Train the model
        model.fit(X_train, Y_train, epochs=epochs, batch_size=6, verbose=0)

        # Evaluate the model on the testing data
        loss = model.evaluate(X_test, Y_test, verbose=0)
        loss_list.append(loss)
        print('Mean Squared Error (MSE) on test data:', loss)

        # Make predictions
        predictions = model.predict(X_test)

        # Plot the predictions and actual values
        plt.figure()
        plt.plot(predictions, color='r', label=f'Predictions (MSE: {loss:.7f})')
        plt.plot(Y_test, color='b', label='Actual')
        plt.xlabel('Data Point Indices')
        plt.ylabel('Residuals [log2(TOF) - Y0(log2(Pass Energy))]')
        # Set y-axis limits to ensure consistent tick marks
        y_min = min(np.min(predictions), np.min(Y_test))
        y_max = max(np.max(predictions), np.max(Y_test))
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.title(f'Model with {epochs} epochs: Residual Method')
        plt.figtext(0.95, 0.05, f'Epochs: {epochs}', ha='right', fontsize=8)

        # Save the plot to the PDF file
        pdf_pages.savefig()
        plt.close()

    # Print the list of MSE for different epochs
    print(loss_list)

    # Close the PDF after saving all the plots
    pdf_pages.close()


def create_model():
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    
    model = Sequential()
    model.add(Dense(9, input_shape=(3,)))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(6))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(2, activation=swish))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return(model)

def convert_to_log(data):
    pass_e = data[1, :]
    tof = data[3, :]
    log_pass = np.log2(pass_e)
    log_tof = np.log2(tof)
    data[1, :] = log_pass
    data[3, :] = log_tof
    return data

def run_model(model_data, epochs=10):
    # Assign X and Y values
    print(model_data.shape)
    data = copy.deepcopy(model_data)
    #data = convert_to_log(data)
    # turn the tof column into a residuals column
    # actual TOF minus TOF for input pass energies
    residual = data[3, :] - y0_NM(data[1, :])
    X = data[0:3, :]
    print(X.shape)
    Y = residual
    # Split the model_data into train and test data
    # Separate the test data
    x, x_test, y, y_test = train_test_split(X.T, Y, test_size=0.15, shuffle=True)
    # Split the remaining data to train and validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, shuffle=True)
    # x_train_tensor = tf.data.Dataset.from_tensor_slices(X_train).batch(BATCH_SIZE)
    # y_train_tensor = tf.data.Dataset.from_tensor_slices(Y_train).batch(BATCH_SIZE)
    # x_test_tensor = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE)
    # y_test_tensor = tf.data.Dataset.from_tensor_slices(Y_test).batch(BATCH_SIZE)

    model = create_model()

    N_TRAIN = x_train.shape[0]
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                        epochs=epochs, validation_data=(x_val, y_val))
    print(f"model history is : \n {history.history}")
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    # get the highest validation accuracy of the training epochs

    fig, ax = plt.subplots()
    epochs = range(1, epochs+1)
    ax.plot(epochs[1:], loss_train[1:], 'g', label='Training loss')
    ax.plot(epochs[1:], loss_val[1:], 'b', label='validation loss')
    ax.set_title('Training and Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    legend0 = ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    print(x_test, y_test)

    test_loss = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest loss:', test_loss)
    """
    # get the highest validation accuracy of the training epochs
    predictions = model.predict(x_test)
    fig, ax1 = plt.subplots()
    ax1.plot(predictions, 'g', label='predictions')
    #ax1.plot(y_test, 'b', label='actual')
    ax1.set_title('prediction and test data')
    ax1.set_xlabel('points')
    ax1.set_ylabel('residual')
    legend1 = ax1.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    """