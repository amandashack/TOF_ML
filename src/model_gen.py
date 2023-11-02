import numpy as np
import os
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU

# Define Swish activation function
def swish(x):
    return x * tf.sigmoid(x)


# Defining the Y0 from the regression line on the NM simulation datasets
def y0_NM(x_value):
    y0 = -0.4821 * x_value - 0.7139
    return y0
