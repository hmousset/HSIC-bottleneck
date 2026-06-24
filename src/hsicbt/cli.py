"""Command-line entry points for the HSIC-Bottleneck experiments.

``hsicbt``       runs a training / experiment task (replaces the old ``run_hsicbt``).
``hsicbt-plot``  renders the figures for a task (replaces the old ``run_plot``).

Both are Hydra applications: every config key can be overridden on the command
line, e.g. ``hsicbt experiment=general-hsicbt learning_rate=0.001 epochs=2``.
"""

import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .config import register_configs
from .core.dispatcher import job_execution, plot_execution
from .utils.color import print_highlight
from .utils.io import ensure_asset_dirs
from .utils.path import get_log_raw_filepath_

register_configs()


def _apply_debug_overrides(config_dict):
    """Honour the HSICBT_DEBUG environment variable (parity with the old CLI).

    The training loops in :mod:`hsicbt.core` also read ``HSICBT_DEBUG == '4'`` to
    stop after a few batches, which makes mode 4 a fast end-to-end smoke test.
    """
    debug_mode = os.environ.get("HSICBT_DEBUG")
    if not debug_mode:
        return
    debug_mode = int(debug_mode)
    if debug_mode == 1:
        config_dict["epochs"] = 1
        config_dict["verbose"] = 1
        print_highlight("debug mode 1: run 1 epoch and print models/configs", "yellow")
    elif debug_mode == 2:
        config_dict["epochs"] = 1
        print_highlight("debug mode 2: run 1 epoch with full batches", "yellow")
    elif debug_mode == 3:
        config_dict["verbose"] = 1
        print_highlight("debug mode 3: print models and configs", "yellow")
    elif debug_mode == 4:
        config_dict["epochs"] = 1
        config_dict["verbose"] = 1
        print_highlight("debug mode 4: run only 1 batch and 1 epoch", "yellow")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_asset_dirs()
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    _apply_debug_overrides(config_dict)
    job_execution(config_dict)


@hydra.main(version_base=None, config_path="conf", config_name="plot_config")
def plot_main(cfg: DictConfig) -> None:
    ensure_asset_dirs()
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict["exp_idx"] = config_dict.get("exp_idx", 1)

    filename = config_dict.pop("filename", "")
    if filename:
        filepath = get_log_raw_filepath_(filename)
        config_dict["filepath"] = filepath
        if not os.path.exists(filepath):
            raise ValueError(f"Invalid path [{filepath}]")
        data_log = np.load(filepath, allow_pickle=True)[()]
        config_dict["task"] = data_log["config_dict"]["task"]
        config_dict["data_code"] = data_log["config_dict"]["data_code"]
        config_dict["training_type"] = data_log["config_dict"]["training_type"]

    plot_execution(config_dict)


if __name__ == "__main__":
    main()
