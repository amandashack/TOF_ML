"""Plugin system for ML-provenance pipeline."""

from src.tof_ml.pipeline.registry import (
    PluginRegistry,
    get_registry
)

__all__ = [
    'PluginRegistry',
    'get_registry'
]