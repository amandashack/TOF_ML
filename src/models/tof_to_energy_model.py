import numpy as np
import re
import os
import datetime
#import fcntl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import glob
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import Callback


"""class MetaFileCallback(tf.keras.callbacks.Callback):
    def __init__(self, param_ID, job_name, meta_file):
        super(MetaFileCallback, self).__init__()
        self.param_ID = param_ID
        self.job_name = job_name
        self.meta_file = meta_file

    def on_epoch_end(self, epoch, logs=None):
        # Update the meta file with the latest checkpoint number (epoch + 1)
        try:
            with open(self.meta_file, 'r+') as f:
                # Acquire exclusive lock
                fcntl.flock(f, fcntl.LOCK_EX)
                lines = f.readlines()
                f.seek(0)
                f.truncate()
                found = False
                for line in lines:
                    line = line.strip()
                    if line.startswith(f"{self.param_ID}|{self.job_name}|"):
                        # Update the line with the latest checkpoint number
                        new_line = f"{self.param_ID}|{self.job_name}|{epoch + 1}"
                        f.write(new_line + '\n')
                        found = True
                    else:
                        f.write(line + '\n')
                if not found:
                    # If the line is not found, add it
                    new_line = f"{self.param_ID}|{self.job_name}|{epoch + 1}"
                    f.write(new_line + '\n')
                # Release the lock
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Error updating meta file: {e}")"""


# BaseModel class using OOP principles
class BaseModel(tf.keras.Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.build_model()

    def build_model(self):
        raise NotImplementedError("Subclasses should implement this method")

    def call(self, inputs):
        raise NotImplementedError("Subclasses should implement this method")

    def get_config(self):
        config = super(BaseModel, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Custom preprocessing layers
class LogTransformLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(LogTransformLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Apply log2 transformation to specific features (e.g., columns 0 and 3)
        log_transformed = tf.concat([
            inputs[:, 0:3],  # Keep features 0,1,2 as is
            tf.math.log(inputs[:, 3:4]) / tf.math.log(2.0)  # Log2 of fourth feature
        ], axis=1)
        return log_transformed

    def get_config(self):
        config = super(LogTransformLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class InteractionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(InteractionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Inputs shape: (batch_size, num_features)
        x1 = inputs[:, 0:1]
        x2 = inputs[:, 1:2]
        x3 = inputs[:, 2:3]
        x4 = inputs[:, 3:4]
        interaction_terms = tf.concat([
            x1 * x2,
            x1 * x3,
            x1 * x4,
            x2 * x3,
            x2 * x4,
            x3 * x4,
            tf.square(x4)
        ], axis=1)
        # Concatenate original inputs with interaction terms
        return tf.concat([inputs, interaction_terms], axis=1)

    def get_config(self):
        config = super(InteractionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ScalingLayer(layers.Layer):
    def __init__(self, min_values, max_values, **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)
        self.min_values = tf.constant(min_values, dtype=tf.float32)
        self.max_values = tf.constant(max_values, dtype=tf.float32)

    def call(self, inputs):
        # Apply Min-Max scaling
        return (inputs - self.min_values) / (self.max_values - self.min_values)

    def get_config(self):
        config = super(ScalingLayer, self).get_config()
        config.update({
            'min_values': self.min_values.numpy().tolist(),
            'max_values': self.max_values.numpy().tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        min_values = np.array(config.pop('min_values'))
        max_values = np.array(config.pop('max_values'))
        return cls(min_values=min_values, max_values=max_values, **config)

# TofToEnergyModel class
class TofToEnergyModel(BaseModel):
    def __init__(self, params, min_values, max_values, **kwargs):
        self.min_values = min_values
        self.max_values = max_values
        self.params = params
        super(TofToEnergyModel, self).__init__(**kwargs)

    def build_model(self):
        layer_size = int(self.params['layer_size'])
        dropout_rate = float(self.params['dropout'])
        optimizer_name = self.params['optimizer']
        learning_rate = float(self.params['learning_rate'])
        job_name = self.params['job_name']

        # Preprocessing layers
        self.preprocessing_layers = self.get_preprocessing_layers()

        # Hidden layers
        self.hidden_layers = []
        if job_name == 'tofs_complex_relu':
            self.hidden_layers.append(layers.Dense(layer_size, activation='relu'))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
            self.hidden_layers.append(layers.Dense(layer_size * 2, activation='relu'))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
            self.hidden_layers.append(layers.Dense(layer_size, activation='relu'))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
        elif job_name == 'tofs_complex_swish':
            self.hidden_layers.append(layers.Dense(layer_size, activation='relu'))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
            self.hidden_layers.append(layers.Dense(layer_size * 2, activation='swish'))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
            self.hidden_layers.append(layers.Dense(layer_size, activation='swish'))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
        elif job_name == "default_deep":
            self.hidden_layers.append(layers.Dense(layer_size, activation='relu'))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
            self.hidden_layers.append(layers.Dense(layer_size * 2, activation='relu'))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
            self.hidden_layers.append(layers.Dense(layer_size * 2, activation='swish'))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
            self.hidden_layers.append(layers.Dense(layer_size, activation='swish'))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
        else:
            print("job name did not match")

        # Output layer
        self.output_layer = layers.Dense(1, activation='linear')

        # Compile the model
        if optimizer_name == "Adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer_name == "RMSprop":
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.compile(loss='mse', optimizer=optimizer, steps_per_execution=self.params['steps_per_execution'])

    def get_preprocessing_layers(self):
        # Define preprocessing layers
        preprocessing_layers = []

        # Log transform layer for specific features
        preprocessing_layers.append(LogTransformLayer())

        # Interaction terms
        preprocessing_layers.append(InteractionLayer())

        # Scaling layer (exclude the output feature from min_values and max_values)
        preprocessing_layers.append(ScalingLayer(self.min_values[1:], self.max_values[1:]))

        return preprocessing_layers

    def call(self, inputs):
        x = inputs
        for layer in self.preprocessing_layers:
            x = layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

    def get_config(self):
        config = super(TofToEnergyModel, self).get_config()
        config.update({
            'params': self.params,
            'min_values': self.min_values.tolist(),
            'max_values': self.max_values.tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        params = config.pop('params')
        min_values = np.array(config.pop('min_values'))
        max_values = np.array(config.pop('max_values'))
        return cls(params, min_values, max_values, **config)
