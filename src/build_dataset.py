import os

from .loaders import DataGenerator

def data_gen(data_filepath, out_path, params, subset_percentage=None):
    # TODO: make this more functional by using the right data loader based on the model chosen
    # Define paths

    scalers_path = os.path.join(out_path, 'scalers.pkl')
    # Load data from the HDF5 file
    data_generator = DataGenerator(data_filepath, batch_size=params['batch_size'])
    data_generator.initialize_data()

    # Optionally select a subset of the training data
    if subset_percentage is not None and 0 < subset_percentage < 1:
        data_generator.subsample_data(subset_percentage)
        # print(f"Original training data size: {original_size}")
        # print(f"Subset training data size: {subset_size}")

    # Split data into train and test sets
    data_generator.partition_data(train_size=0.8)

    # Save test data
    data_generator.save_test_data(out_path)

    # Calculate or load scalers
    data_generator.calculate_scalers(scalers_path)

    return data_generator
