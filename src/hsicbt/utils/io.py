import os

import numpy as np
import torch
import yaml

from .color import print_highlight
from .path import attaching_timestamp_filepath, make_symlink

ASSET_DIRS = [
    "./assets",
    "./assets/data",
    "./assets/logs",
    "./assets/logs/raw",
    "./assets/exp",
    "./assets/exp/raw",
    "./assets/models",
    "./assets/models/raw",
    "./assets/tmp",
    "./assets/activation",
    "./assets/activation/raw",
]


def ensure_asset_dirs():
    """Create the assets/* directory tree used for logs, models and figures."""
    for path in ASSET_DIRS:
        os.makedirs(path, exist_ok=True)


def load_yaml(filepath):
    with open(filepath) as stream:
        try:
            data = yaml.load(stream, yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    print_highlight(f"Loaded  [{filepath}]")
    return data


def save_model(model, filepath, sympath):
    timestamp_path = attaching_timestamp_filepath(filepath)
    torch.save(model.state_dict(), timestamp_path)
    print_highlight(f"Saved   [{timestamp_path}]")
    make_symlink(timestamp_path, sympath)


def load_model(filepath):
    model = torch.load(filepath)
    print_highlight(f"Loaded  [{filepath}]")
    return model


def save_logs(logs, filepath):
    timestamp_path = attaching_timestamp_filepath(filepath)
    np.save(timestamp_path, logs)
    make_symlink(timestamp_path, filepath)
    print_highlight(f"Saved   [{timestamp_path}]", ctype="blue")


def load_logs(filepath):
    logs = np.load(filepath, allow_pickle=True)[()]
    print_highlight(f"Loaded  [{filepath}]", ctype="blue")
    return logs
