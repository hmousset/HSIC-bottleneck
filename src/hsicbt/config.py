"""Structured Hydra configuration schemas for the HSIC-Bottleneck experiments.

The dataclasses below describe the full key surface that the legacy YAML configs
used to carry. Every field has a default so that experiment configs only need to
override what differs, and so the plain ``dict`` handed to the dispatcher always
contains the keys the training/plotting code expects.
"""

from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class ExperimentConfig:
    # optimisation
    batch_size: int = 256
    learning_rate: float = 0.001
    n_layers: int = 5
    hidden_width: int = 128
    last_hidden_width: int | None = None
    epochs: int = 5
    seed: int = 1234

    # HSIC objective
    sigma: float = 5.0
    lambda_x: float = 1.0
    lambda_y: float = 50.0

    # model / dataset
    device: str = "cuda"
    model: str = "linear"
    atype: str = "relu"
    data_code: str = "mnist"
    model_file: str = "hsic_weight_linear_mnist.pt"
    checkpoint: int | None = None

    # task control
    task: str = "hsic-train"
    training_type: str = "hsictrain"
    do_training: bool = True
    exp_index: Any = None

    # logging / output
    log_batch_interval: int = 10
    verbose: int = 0
    ext: str = "pdf"


@dataclass
class PlotConfig:
    task: str = "general"
    data_code: str = "mnist"
    ext: str = "pdf"
    training_type: str = "hsictrain"
    footnote: str = ""
    exp_idx: int = 1
    plot_task: int = 1
    # optional: a specific raw log filename under assets/logs/raw to plot
    filename: str = ""


def register_configs() -> None:
    """Register the structured schemas so Hydra can validate overrides."""
    cs = ConfigStore.instance()
    cs.store(name="experiment_schema", node=ExperimentConfig)
    cs.store(name="plot_schema", node=PlotConfig)
