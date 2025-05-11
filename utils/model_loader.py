import yaml

def load_model_names(config_path: str = "./configs/model_names.yaml") -> list:
    """Load model names from a YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["models"]