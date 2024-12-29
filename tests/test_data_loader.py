import os
import numpy as np
import pytest
from src.data.csv_loader import CSVDataLoader

def test_csv_loader():
    # Arrange
    test_data_path = "test_data/input_data"
    # Make sure test_data_path has some mock CSV files, e.g. NM_neg_0.csv with known contents.

    loader = CSVDataLoader()

    # Act
    data = loader.load_data(folder_path=test_data_path)

    # Assert
    assert isinstance(data, np.ndarray), "Data should be a NumPy array"
    assert data.shape[1] == 4, "Data should have 4 columns: final_TOF, retardation, initial_KE, final_X"
    # You can add more assertions based on expected values or known tests.

    # Example: Check that final_X column is greater than 406 as per the masking condition
    assert np.all(data[:, -1] > 406), "All final_X values should be greater than 406 due to masking"
