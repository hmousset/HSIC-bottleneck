#!/bin/bash

# Grid search for the unformat (hsic-solve) experiment.
p_sigma=(1 2)
p_batchsize=512
p_lambda=(100 200)
p_learningrate=(0.005)
p_depth=5
n_epoch=1
t_type=hsictrain  # backprop/hsictrain
t_data=fmnist

for p_s in "${p_sigma[@]}"; do
    for p_l in "${p_lambda[@]}"; do
        for p_lr in "${p_learningrate[@]}"; do
            echo -e "\e[1m\e[100m ======> ${t_data}; ${t_type}; sigma: ${p_s}, lambda: ${p_l}," \
                 "batch_size: ${p_batchsize}, learning rate: ${p_lr}, n_epoch: ${n_epoch}," \
                 "n_layers: ${p_depth} \e[0m"
            uv run hsicbt experiment=hsicsolve-beta \
                training_type=${t_type} epochs=${n_epoch} sigma=${p_s} lambda_y=${p_l} \
                learning_rate=${p_lr} data_code=${t_data} batch_size=${p_batchsize} n_layers=${p_depth}
        done
    done
done
