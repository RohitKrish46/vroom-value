import yaml

def load_hyperparameters(config_path: str = "./configs/hyperparameter.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config