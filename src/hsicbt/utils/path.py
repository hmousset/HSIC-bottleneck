import os

from .color import print_highlight
from .const import TIMESTAMP_CODE


def attaching_timestamp_filepath(filepath):
    filename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    filename, ext = os.path.splitext(filename)
    filename_time = f"{TIMESTAMP_CODE}_{filename}{ext}"
    timestamp_path = os.path.join(dirname, "raw", filename_time)
    return timestamp_path


def make_symlink(src_path, sym_path):
    if os.path.exists(sym_path):
        os.remove(sym_path)
    os.symlink(src_path, sym_path)
    print_highlight(f"Symlink [{sym_path}]", ctype="blue")


def code_name(task, ttype, dtype, idx):
    if idx:
        filename = f"{task}-{ttype}-{dtype}-{idx:04d}.npy"
    else:
        filename = f"{task}-{ttype}-{dtype}.npy"
    return filename


def get_log_filepath(task, ttype, dtype, idx=None):
    return f"{os.getcwd()}/assets/logs/{code_name(task, ttype, dtype, idx)}"


def get_log_raw_filepath(task, ttype, dtype, idx=None):
    return f"{os.getcwd()}/assets/raw/logs/{code_name(task, ttype, dtype, idx)}"


def get_log_raw_filepath_(filename):
    return f"{os.getcwd()}/assets/logs/raw/{filename}"


def get_plot_filename(config_dict):
    return "{}-{}".format(config_dict["task"], config_dict["data_code"])


def get_exp_path(filename):
    return f"{os.getcwd()}/assets/exp/{filename}"


def get_exp_raw_path(filename):
    return f"{os.getcwd()}/assets/exp/raw/{filename}"


def get_act_path(task, ttype, dtype, idx=None):
    return f"{os.getcwd()}/assets/activation/{code_name(task, ttype, dtype, idx)}"


def get_act_raw_path(task, ttype, dtype, idx=None):
    return f"{os.getcwd()}/assets/activation/raw/{code_name(task, ttype, dtype, idx)}"


def get_act_raw_path_(filename, idx=""):
    if idx:
        idx = f"-{idx:04d}"
    return f"{os.getcwd()}/assets/activation/raw/{filename[:-4]}{idx}.{filename[-3:]}"


def get_model_path(filename, idx=None):
    if idx:
        return f"{os.getcwd()}/assets/models/{os.path.splitext(filename)[0]}-{idx:04d}.pt"
    return f"{os.getcwd()}/assets/models/{filename}"


def get_model_raw_path(filename, idx=None):
    if idx:
        return f"{os.getcwd()}/assets/models/raw/{os.path.splitext(filename)[0]}-{idx:04d}.pt"
    return f"{os.getcwd()}/assets/models/raw/{filename}"


def get_tmp_path(filename):
    return f"{os.getcwd()}/assets/tmp/{filename}"
