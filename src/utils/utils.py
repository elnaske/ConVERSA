import yaml
import json

def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def save_to_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f)