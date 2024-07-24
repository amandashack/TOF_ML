import tensorflow as tf
import numpy as np

def evaluate(model, test_dataset, batch_size=1024):
    """
    Evaluate the model on test data.

    Parameters:
    model: The trained model.
    test_inputs: Inputs of the test dataset.
    test_outputs: Outputs of the test dataset.
    batch_size: Size of the batches for evaluation.

    Returns:
    test_loss: Loss on the test dataset.
    """
    print("Starting evaluation...")  # Debugging statement
    # Calculate the number of steps
    steps = np.ceil(len(test_dataset) / batch_size).astype(int)
    loss = model.evaluate(test_dataset, steps=steps)
    print("Evaluation completed.")  # Debugging statement
    return loss
