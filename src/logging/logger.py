import logging.config
import yaml


def setup_logger(name: str, config_path="config/logging_config.yaml"):
    with open(config_path, 'r') as f:
        log_config = yaml.safe_load(f)
    logging.config.dictConfig(log_config)
    return logging.getLogger(name)
