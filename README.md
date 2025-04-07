# ML Provenance Tracker Framework

A generalized, plugin-based machine learning framework with comprehensive provenance tracking.

## Overview

This framework provides a flexible, modular approach to building and tracking machine learning workflows. It enables:

- Tracking data lineage and transformations
- Using custom data loaders for different data formats
- Defining reproducible preprocessing pipelines
- Training models with customizable hyperparameters
- Automatic reporting and visualization
- Integration with database for experiment tracking

The system is designed with pluggable components that can be configured via YAML files or command-line options, making it adaptable to various data sources and model types.

## Key Features

- **Plugin Architecture**: Easily extend with custom data loaders, transformers, splitters, and models
- **Provenance Tracking**: Complete lineage tracking of data from source to final model
- **Configuration-Driven**: Control workflow with YAML configuration files
- **Command-Line Interface**: Run workflows from the command line with parameter overrides
- **Comprehensive Reporting**: Automated reports for data, training, and evaluation
- **Database Integration**: Track experiments and compare results
- **Visualization Tools**: Visualize data distributions, model performance, and data lineage

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-provenance-tracker.git
cd ml-provenance-tracker

# Install dependencies
pip install -e .
```

### Basic Usage

1. Configure your pipeline in `config/base_config.yaml`
2. Run the pipeline:

```bash
python -m tof_ml.cli --config config/base_config.yaml
```

### Command-Line Options

Override configuration settings via the command line:

```bash
python -m tof_ml.cli --config config/base_config.yaml \
    --mode train \
    --data-loader H5Loader \
    --dataset-path data/my_dataset.h5 \
    --model-type MLPKerasRegressor \
    --epochs 100 \
    --batch-size 64
```

## Configuration

The framework uses YAML configuration files:

- `base_config.yaml`: Main configuration file
- `class_mapping_config.yaml`: Maps component types to Python classes
- `database_config.yaml`: Database connection settings

Example configuration:

```yaml
experiment_name: "my_experiment"
output_dir: "./output"

data:
  loader_config_key: "H5Loader"
  dataset_path: "./data/dataset.h5"
  feature_columns: ["tof", "retardation", "amplitude"]
  target_columns: ["energy"]

preprocessing:
  transformers:
    - type: "Normalizer"
      columns: ["tof", "amplitude"]
      method: "standard"

model:
  type: "MLPKerasRegressor"
  hidden_layers: [64, 128, 64]
  activations: ["relu", "relu", "relu"]
  epochs: 100
  batch_size: 32
```

## Extending the Framework

### Adding a Custom Data Loader

1. Create a new loader class:

```python
# src/tof_ml/data/loaders/my_custom_loader.py
class MyCustomLoader:
    def __init__(self, config, **kwargs):
        self.config = config
        # Initialize your loader
        
    def load_data(self):
        # Implement your loading logic
        return loaded_data
```

2. Add it to the class mapping in `config/class_mapping_config.yaml`:

```yaml
Loader:
  MyCustomLoader: "tof_ml.data.loaders.my_custom_loader.MyCustomLoader"
```

3. Use it in your configuration:

```yaml
data:
  loader_config_key: "MyCustomLoader"
  # Custom loader parameters
```

### Adding a Custom Model

1. Create a new model class:

```python
# src/tof_ml/models/my_custom_model.py
class MyCustomModel:
    def __init__(self, **kwargs):
        # Initialize your model
        
    def fit(self, X, y, **kwargs):
        # Train your model
        return history
    
    def predict(self, X):
        # Make predictions
        return predictions
```

2. Add it to the class mapping:

```yaml
Model:
  MyCustomModel: "tof_ml.models.my_custom_model.MyCustomModel"
```

3. Use it in your configuration:

```yaml
model:
  type: "MyCustomModel"
  # Custom model parameters
```

## Provenance Tracking

The framework automatically tracks data provenance:

- Data sources and their metadata
- Transformations applied to data
- Data splits
- Model training inputs and parameters
- Evaluation results

View the provenance graph:

```python
from tof_ml.data.data_provenance import ProvenanceTracker

tracker = ProvenanceTracker(config)
tracker.visualize_provenance_graph("provenance_graph.svg")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
