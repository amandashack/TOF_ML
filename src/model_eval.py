import tensorflow as tf


def evaluate(model, x_test, y_test):
    test_loss = model.evaluate(x_test, y_test, verbose=2)
    return test_loss
