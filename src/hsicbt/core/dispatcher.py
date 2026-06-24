import torch

from ..task.task_general import plot_general_result, task_general_func
from ..task.task_hsicsolve import plot_hsicsolve_result, task_hsicsolve_func
from ..task.task_needle import plot_needle_result, task_needle_func
from ..task.task_resconv import plot_resconv_result, task_resconv_func
from ..task.task_sigmacomb import plot_sigmacomb_result, task_sigmacomb_func
from ..task.task_variedact import plot_variedact_result, task_variedact_func
from ..task.task_varieddepth import plot_varieddepth_result, task_varieddepth_func
from ..task.task_varieddim import plot_varieddim_result, task_varieddim_func
from ..task.task_variedep import plot_variedep_result, task_variedep_func
from .engine import training_format, training_hsic, training_standard


def plot_execution(config_dict):

    if config_dict["task"] == "standard-train":
        if config_dict["do_training"]:
            out_batch, out_epoch = training_standard(config_dict)

    elif config_dict["task"] == "hsic-train":
        if config_dict["do_training"]:
            out_batch, out_epoch = training_hsic(config_dict)

    elif config_dict["task"] == "format-train":
        if config_dict["do_training"]:
            out_batch, out_epoch = training_format(config_dict)

    elif config_dict["task"] == "needle":
        plot_needle_result(config_dict)

    elif config_dict["task"] == "general":
        plot_general_result(config_dict)

    elif config_dict["task"] == "hsic-solve":
        plot_hsicsolve_result(config_dict)

    elif config_dict["task"] == "varied-activation":
        plot_variedact_result(config_dict)

    elif config_dict["task"] == "sigma-combined":
        plot_sigmacomb_result(config_dict)

    elif config_dict["task"] == "varied-depth":
        plot_varieddepth_result(config_dict)

    elif config_dict["task"] == "varied-epoch":
        plot_variedep_result(config_dict)

    elif config_dict["task"] == "varied-dim":
        plot_varieddim_result(config_dict)

    elif config_dict["task"] == "resconv":
        plot_resconv_result(config_dict)

    else:
        raise ValueError(
            "Unknown given task [{}], please check \
            hsicbt.dispatcher.job_execution".format(config_dict["task"])
        )


def job_execution(config_dict):

    torch.cuda.manual_seed(config_dict["seed"])
    torch.manual_seed(config_dict["seed"])
    if config_dict["task"] == "standard-train":
        if config_dict["do_training"]:
            out_batch, out_epoch = training_standard(config_dict)

    elif config_dict["task"] == "hsic-train":
        if config_dict["do_training"]:
            out_batch, out_epoch = training_hsic(config_dict)

    elif config_dict["task"] == "format-train":
        if config_dict["do_training"]:
            out_batch, out_epoch = training_format(config_dict)

    elif config_dict["task"] == "needle":
        task_needle_func(config_dict)

    elif config_dict["task"] == "general":
        task_general_func(config_dict)

    elif config_dict["task"] == "hsic-solve":
        task_hsicsolve_func(config_dict)

    elif config_dict["task"] == "varied-activation":
        task_variedact_func(config_dict)

    elif config_dict["task"] == "sigma-combined":
        task_sigmacomb_func(config_dict)

    elif config_dict["task"] == "varied-depth":
        task_varieddepth_func(config_dict)

    elif config_dict["task"] == "varied-epoch":
        task_variedep_func(config_dict)

    elif config_dict["task"] == "varied-dim":
        task_varieddim_func(config_dict)

    elif config_dict["task"] == "resconv":
        task_resconv_func(config_dict)

    else:
        raise ValueError(
            "Unknown given task [{}], please check \
            hsicbt.dispatcher.job_execution".format(config_dict["task"])
        )
