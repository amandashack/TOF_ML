# Test the experiment ID generation:
from src.tof_ml.database.api import DBApi


def test_experiment_id_generation():
    """Test the experiment ID generation function."""

    test_cases = [
        "MNIST CNN Classification",
        "mnist_cnn_classification",
        "Deep Learning Experiment #1",
        "Image Classification - ResNet50",
        "NLP Transformer Model",
        "Time Series Forecasting"
    ]

    print("Testing Experiment ID Generation:")
    print("=" * 50)

    for name in test_cases:
        experiment_id = DBApi.generate_experiment_id(name)
        slug = DBApi.slugify(name)
        print(f"Name: '{name}'")
        print(f"Slug: '{slug}'")
        print(f"ID:   '{experiment_id}'")
        print("-" * 30)

# Expected output for "MNIST CNN Classification":
# Name: 'MNIST CNN Classification'
# Slug: 'mnist-cnn-classification'
# ID:   'mnist-cnn-classification-5f8b7a'

# Benefits of this approach:
# 1. Stable - same name always generates same ID
# 2. Readable - includes human-readable slug
# 3. Unique - hash suffix prevents collisions
# 4. URL-friendly - can be used in web interfaces
# 5. Consistent - deterministic generation

test_experiment_id_generation()