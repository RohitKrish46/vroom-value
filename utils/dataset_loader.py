import yaml

def load_data_path(config_path: str = "./configs/dataset.yaml") -> str:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["dataset"]