import yaml


def load_models(path):

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config["models"]