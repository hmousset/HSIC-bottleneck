#!/bin/bash

# Grid search over sigma (and other knobs). Every config key is overridable on
# the command line; see src/hsicbt/config.py for the full list.
p_sigma=(5 6 7 8)
p_batchsize=256
p_lambda=(100)
p_learningrate=(0.0005)
p_depth=5
n_epoch=1
t_type=hsictrain  # backprop/hsictrain
t_data=mnist

for p_s in "${p_sigma[@]}"; do
    for p_l in "${p_lambda[@]}"; do
        for p_lr in "${p_learningrate[@]}"; do
            echo -e "\e[1m\e[100m ======> ${t_data}; sigma: ${p_s}, lambda: ${p_l}," \
                 "batch_size: ${p_batchsize}, learning rate: ${p_lr}, n_epoch: ${n_epoch}," \
                 "n_layers: ${p_depth} \e[0m"
            uv run hsicbt experiment=general-hsicbt \
                training_type=${t_type} epochs=${n_epoch} sigma=${p_s} lambda_y=${p_l} \
                learning_rate=${p_lr} data_code=${t_data} batch_size=${p_batchsize} n_layers=${p_depth}
            uv run hsicbt experiment=general-format epochs=1
        done
    done
done
