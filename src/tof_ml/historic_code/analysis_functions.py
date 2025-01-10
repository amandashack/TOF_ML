import pickle
import io
from PyPDF2 import PdfReader, PdfWriter
import numpy as np
import h5py
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
from loaders.load_and_save import DataGenerator, DataGeneratorWithVeto

def fig_to_pdf_page(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf')
    buf.seek(0)
    return PdfReader(buf).pages[0]


def load_scalers(scalers_path):
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers


def plot_histograms(data, scalers, sample_size=1000):
    # Randomly sample the data
    sampled_data = data[np.random.choice(data.shape[0], sample_size, replace=False), :]
    mask = sampled_data[:, 7].astype(bool)  # Use the mask to filter the data
    sampled_data = sampled_data[mask]

    # Calculate interaction terms and apply scaling
    generator = DataGenerator(sampled_data, scalers, batch_size=sample_size)
    data_with_interactions = generator.calculate_interactions(sampled_data[:, :5])
    scaled_data = generator.scale_input(data_with_interactions)

    # Create DataFrames for easier manipulation and plotting
    columns = [
        'initial_ke', 'elevation', 'retardation', 'mid1', 'mid2',
        'initial_ke*elevation', 'initial_ke*mid1', 'initial_ke*mid2',
        'elevation*retardation', 'elevation*mid1', 'elevation*mid2',
        'retardation*mid1', 'retardation*mid2',
        'initial_ke^2', 'elevation^2', 'retardation^2', 'mid1^2', 'mid2^2'
    ]
    df_unscaled = pd.DataFrame(data_with_interactions, columns=columns)
    df_scaled = pd.DataFrame(scaled_data, columns=columns)

    # Plot histograms for each column
    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        sns.histplot(df_unscaled[col], bins=30, kde=True, color='skyblue', edgecolor='k', ax=axes[0])
        axes[0].set_title(f'Unscaled {col}')
        sns.histplot(df_scaled[col], bins=30, kde=True, color='salmon', edgecolor='k', ax=axes[1])
        axes[1].set_title(f'Scaled {col}')
        plt.tight_layout()
        plt.show()

def load_test_data(test_data_path):
    with h5py.File(test_data_path, 'r') as hf:
        test_data = hf['test_data'][:]
    return test_data


# Function to randomly sample data
def random_sample_data(data, sample_size):
    indices = np.random.choice(data.shape[0], min(sample_size, data.shape[0]), replace=False)
    return data[indices]


def check_and_clean_data(data):
    # Replace infinities and NaNs with a large negative value
    data = np.nan_to_num(data, nan=-1e10, posinf=1e10, neginf=-1e10)
    return data

