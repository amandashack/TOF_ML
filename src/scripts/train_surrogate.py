import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.model_selection import KFold
import re
import sys
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import glob
from scripts import preprocess_surrogate_test_data, evaluate, partition_data, save_test_data
from loaders import DataGenerator, DataGeneratorWithVeto
from models import train_main_model, create_main_model
from models import train_veto_model
import logging

tf.get_logger().setLevel(logging.ERROR)

DATA_FILENAME = r"/sdf/home/a/ajshack/combined_data_shuffled.h5"
VETO_MODEL = r"/sdf/home/a/ajshack/TOF_ML/stored_models/surrogate/veto_model.h5"

# Check GPU availability and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus = None
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        pass
        #print(e)

def calculate_scalers(data, scalers_path):
    if os.path.exists(scalers_path):
        # Load scalers from file
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        print(f"Scalers loaded from {scalers_path}")
    else:
        # Calculate interaction terms for the entire data
        generator = DataGenerator(data, None)
        data_with_interactions = generator.calculate_interactions(data[:, :5])
        all_data = np.column_stack([data_with_interactions, data[:, 5:7]])
        scalers = MinMaxScaler()
        scalers.fit(all_data)

        # Save scalers
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to {scalers_path}")

    return scalers

def run_train(out_path, params, n_splits=3, subset_percentage=None):
    checkpoint_dir = os.path.join(out_path, "checkpoints")
    combined_model_path = os.path.join(out_path, "combined_model.h5")
    # Load data from the HDF5 file
    with h5py.File(DATA_FILENAME, 'r') as hf:
        # preprocess data
        data = hf["data"][:]
        # log of TOF and log of KE
        data[:, 0] = np.log(data[:, 0] + data[:, 2])  # initial kinetic energy (was in pass energy)
        data[:, 5] = np.log(data[:, 5])

    # Split data into train and test sets
    train_data, test_data = partition_data(data)

    # Optionally select a subset of the training data
    if subset_percentage is not None and 0 < subset_percentage < 1:
        original_size = len(train_data)
        subset_size = int(original_size * subset_percentage)
        train_data = train_data[np.random.choice(original_size, subset_size, replace=False)]
        #print(f"Original training data size: {original_size}")
        #print(f"Subset training data size: {subset_size}")

    # Save test data
    save_test_data(test_data, out_path)

    # Define the path for scalers
    scalers_path = os.path.join(out_path, 'scalers.pkl')

    # Calculate or load scalers
    scalers = calculate_scalers(train_data, scalers_path)

    # Define batch size
    batch_size = int(params["batch_size"])

    # Prepare cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_models = []
    fold = 1

    for train_index, val_index in kfold.split(train_data):
        train_fold_data = train_data[train_index]
        val_fold_data = train_data[val_index]

        # Calculate steps per epoch
        params['steps_per_epoch'] = np.ceil(len(train_fold_data) / batch_size).astype(int)
        params['validation_steps'] = np.ceil(len(val_fold_data) / batch_size).astype(int)

        # Check if veto model exists
        # veto_model = load_veto_model_if_exists(checkpoint_dir, fold)
        veto_model = tf.keras.models.load_model(VETO_MODEL)

        if veto_model is None:
            # If no veto model exists, create generator instances and train a new veto model
            veto_train_gen = DataGenerator(train_fold_data, scalers, batch_size=batch_size)
            veto_val_gen = DataGenerator(val_fold_data, scalers, batch_size=batch_size)

            veto_train_dataset = tf.data.Dataset.from_generator(
                veto_train_gen, output_signature=(
                    tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # mask as the target
                )
            ).take(len(train_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

            veto_val_dataset = tf.data.Dataset.from_generator(
                veto_val_gen, output_signature=(
                    tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # mask as the target
                )
            ).take(len(val_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

            #print(f"Training veto model on fold {fold}/{n_splits}...")
            veto_model, history = train_veto_model(veto_train_dataset, veto_val_dataset, params, checkpoint_dir)
            veto_model_path = os.path.join(checkpoint_dir, f"veto_model_fold_{fold}.h5")
            veto_model.save(veto_model_path)

        # Create generator instances for training and validation datasets using veto model
        train_gen = DataGeneratorWithVeto(train_fold_data, scalers, veto_model, batch_size=batch_size)
        val_gen = DataGeneratorWithVeto(val_fold_data, scalers, veto_model, batch_size=batch_size)

        # Create tf.data.Dataset from the generator
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, output_signature=(
                tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
            )
        ).take(len(train_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_gen, output_signature=(
                tf.TensorSpec(shape=(None, 18), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
            )
        ).take(len(val_fold_data)).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)

        # Train the main model
        #print(f"Training main model on fold {fold}/{n_splits}...")
        model, history = train_main_model(train_dataset, val_dataset, params, checkpoint_dir)

        model.save(os.path.join(out_path, f"main_model_fold_{fold}.h5"))
        with open(os.path.join(out_path, f"history_fold_{fold}.pkl"), 'wb') as f:
            pickle.dump(history.history, f)

        print(f"Models and history for fold {fold} saved.")
        fold_models.append(model)
        fold += 1

        # Combine models by averaging their weights
    combined_model = create_main_model()  # Use your model creation function
    combined_weights = [model.get_weights() for model in fold_models]
    new_weights = []

    for weights_tuple in zip(*combined_weights):
        new_weights.append([np.mean(np.array(w), axis=0) for w in zip(*weights_tuple)])

    combined_model.set_weights(new_weights)
    combined_model.save(combined_model_path)

    # Final evaluation on test data
    x_test, y_test = preprocess_surrogate_test_data(test_data, scalers, veto_model, combined_model)
    loss_test = evaluate(combined_model, x_test, y_test, plot=False)

    print(f"Final test_loss: {loss_test:.4f}")