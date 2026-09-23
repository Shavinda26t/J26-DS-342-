"""Helper utilities for logging, path resolution, and YAML loading."""
import yaml
import os

def load_yaml_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}
