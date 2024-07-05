import tensorflow as tf


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
    #test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_outputs)).batch(batch_size)
    print("Test dataset created.")  # Debugging statement
    loss = model.evaluate(test_dataset)
    print("Evaluation completed.")  # Debugging statement
    return loss
