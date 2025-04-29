import yaml

def load_hyperparameters(config_path: str = "configs/hyperparameters.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config