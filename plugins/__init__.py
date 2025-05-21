"""Plugin system for ML-provenance pipeline."""

from src.tof_ml.pipeline.plugins.interfaces import (
    DataLoaderPlugin,
    ModelPlugin,
    ReportGeneratorPlugin
)
from src.tof_ml.pipeline.plugins.registry import (
    PluginRegistry,
    get_registry
)

__all__ = [
    'DataLoaderPlugin',
    'ModelPlugin',
    'ReportGeneratorPlugin',
    'PluginRegistry',
    'get_registry'
]