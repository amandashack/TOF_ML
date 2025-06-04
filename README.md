# ML Provenance Pipeline with Plugin Architecture

This project implements a modular machine learning pipeline with comprehensive provenance tracking. The pipeline is designed using a plugin architecture that allows easy swapping of components (data loaders, models, report generators) for different machine learning tasks.

## Architecture Overview

The ML Provenance Pipeline uses a plugin-based architecture organized as follows:

1. **Plugin Interfaces**: Abstract base classes defining the contract for each plugin type
2. **Plugin Registry**: Central registry for managing and retrieving plugin implementations
3. **Plugin Implementations**: Concrete implementations for specific tasks (e.g., MNIST)
4. **Pipeline Orchestrator**: Coordinates the execution of the pipeline using the appropriate plugins

## Plugin Types

The pipeline supports the following plugin types:

- **Data Loaders**: Handle loading and preprocessing data from various sources
- **Models**: Provide model implementations for training and inference
- **Report Generators**: Generate reports and visualizations for model performance

## Directory Structure

```
ml_provenance/
├── src/
│   └── tof_ml/
│       ├── data/
│       │   ├── loaders/
│       │   │   └── base_loader.py      # BaseDataLoader + mixins
│       │   ├── data_manager.py         # Handles data operations
│       │   └── data_provenance.py      # Tracks data lineage
│       ├── models/
│       │   └── base_model.py           # BaseModel + mixins
│       ├── training/
│       │   └── trainer.py              # Model training logic
│       ├── reporting/
│       │   └── base_report_generator.py # BaseReportGenerator + mixins
│       ├── database/
│       │   └── api.py                  # Database operations
│       └── pipeline/
│           ├── orchestrator.py         # Coordinates pipeline execution
│           └── plugins/
│               └── registry.py         # Plugin registration
├── examples/
│   └── mnist_hyperparam/              # Example MNIST implementation
├── config/
│   ├── class_mapping_config.yaml     # Plugin class mappings
│   └── database_config.yaml          # Database configuration
└── main.py                    # CLI entry point
```

## Using the Pipeline

### 1. Configure the Pipeline

Create a configuration file for your specific task:

```yaml
# config/your_task_config.yaml

# Experiment configuration
experiment_name: "your_task_name"
output_dir: "./output"

# Plugin configuration
plugins:
  data_loader: "YourDataLoader"
  model: "YourModel"
  report_generator: "YourReportGenerator"

# Loader-specific configuration
data_loader:
  # Your data loader parameters here
  
# Model-specific configuration
model:
  # Your model parameters here
  
# Report configuration
reporting:
  # Your reporting parameters here
```

### 2. Register Your Plugins

Update the class mapping configuration to include your plugins:

```yaml
# config/class_mapping_config.yaml

# Data Loaders
Loader:
  YourDataLoader: "plugins.loaders.your_data_loader.YourDataLoader"
  
# Models
Model:
  YourModel: "plugins.models.your_model.YourModel"
  
# Report Generators
ReportGenerator:
  YourReportGenerator: "plugins.report_generators.your_report.YourReportGenerator"
```

### 3. Run the Pipeline

Run the pipeline using the main script:

```bash
python main.py --config config/your_task_config.yaml
```

## Creating Custom Plugins

### Data Loader Plugin

```python
from src.tof_ml.data.base_data_loader import BaseDataLoaderPlugin
from src.tof_ml.pipeline.plugins.interfaces import DataLoaderPlugin

class YourDataLoader(BaseDataLoaderPlugin, DataLoaderPlugin):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # Your initialization here
        
    def load_data(self):
        # Implement data loading logic
        
    def extract_features_and_targets(self, data=None):
        # Implement feature/target extraction
```

### Model Plugin

```python
from src.tof_ml.models.base_model import BaseModelPlugin

class YourModel(BaseModelPlugin):
    def __init__(self, **kwargs):
        # Your initialization here
        
    def fit(self, X, y, **kwargs):
        # Implement training logic
        
    def predict(self, X):
        # Implement prediction logic
        
    def evaluate(self, X, y, **kwargs):
        # Implement evaluation logic
        
    def save(self, path):
        # Implement model saving
        
    def custom_metrics(self, y_true, y_pred):
        # Implement custom metrics calculation
```

### Report Generator Plugin

```python
from src.tof_ml.reporting.report_generator import ReportGeneratorPlugin

class YourReportGenerator(ReportGeneratorPlugin):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Your initialization here
        
    def generate_data_report(self):
        # Implement data report generation
        
    def generate_training_report(self):
        # Implement training report generation
        
    def generate_evaluation_report(self):
        # Implement evaluation report generation
```

## Example: MNIST Plugin

The project includes an MNIST plugin implementation as an example:

1. **MNISTLoader**: Loads the MNIST dataset from TensorFlow
2. **MNISTKerasModel**: Simple neural network for MNIST classification
3. **MNISTReportGenerator**: Generates MNIST-specific reports and visualizations

Run the MNIST example:

```bash
python main.py --config config/mnist_config.yaml
```

## Extending the Framework

To add support for new ML tasks:

1. Create task-specific plugin implementations in the `plugins/` directory
2. Register them in the class mapping configuration
3. Create a task-specific configuration file
4. Run the pipeline with your configuration

The plugin architecture makes it easy to add new components while reusing the core pipeline infrastructure.