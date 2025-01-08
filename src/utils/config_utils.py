# src/utils/config_loader.py
import yaml
import os

def load_config(config_path: str) -> dict:
    """
    Loads a YAML config file from the given path and returns a Python dictionary.

    :param config_path: Path to the YAML configuration file.
    :return: Dictionary of config parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
