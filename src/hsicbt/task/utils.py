import os

from ..core.engine import training_format, training_hsic, training_standard
from ..utils import plot
from ..utils.color import print_highlight
from ..utils.const import TIMESTAMP_CODE, TTYPE_FORMAT, TTYPE_HSICTRAIN, TTYPE_STANDARD
from ..utils.path import get_exp_path, get_exp_raw_path


def task_assigner(ttype):
    func = None
    if ttype == TTYPE_HSICTRAIN:
        func = training_hsic
    elif ttype == TTYPE_STANDARD:
        func = training_standard
    elif ttype == TTYPE_FORMAT:
        func = training_format
    return func


def get_experiment_fig_filename(config_dict, etype, idx=""):
    filepath = ""
    if idx:
        idx = f"-{idx:04d}"

    if etype == "needle-1":
        filepath = get_exp_path(
            "fig8b-needle-1d-dist-{}{}.{}".format(
                config_dict["training_type"], idx, config_dict["ext"]
            )
        )
    elif etype == "needle-2":
        filepath = get_exp_path(
            "fig8a-needle-1d-dist-{}{}.{}".format(
                config_dict["training_type"], idx, config_dict["ext"]
            )
        )
    return filepath


def save_experiment_fig(filepath):
    filename = os.path.basename(filepath)
    filename_noext, ext = os.path.splitext(filename)
    timestamp_filename = f"{TIMESTAMP_CODE}-{filename_noext}{ext}"
    timestamp_filepath = get_exp_raw_path(timestamp_filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    plot.save_figure(timestamp_filepath)
    os.symlink(timestamp_filepath, filepath)
    print_highlight(f"Symlink [{filepath}]", "blue")
