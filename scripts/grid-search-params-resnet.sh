#!/bin/bash

# Grid search for the ResNet (resconv) experiment: HSIC-training followed by a
# short format-training evaluation. See src/hsicbt/config.py for tunable keys.
p_sigma=(50 100 150 200)
p_lambda=(1000)         # beta, the hsic-bt weight on HSIC(Z_i, Y)
p_learningrate=(0.001)

p_batchsize=512
p_depth=5               # number of hidden layers
n_epoch=1               # number of hsic-training epochs
n_kernels=64            # number of kernels at each layer
t_data=cifar10

for p_s in "${p_sigma[@]}"; do
    for p_l in "${p_lambda[@]}"; do
        for p_lr in "${p_learningrate[@]}"; do
            echo -e "\e[1m\e[100m======> ${t_data}, sigma: ${p_s}, lambda: ${p_l}," \
                 "batch_size: ${p_batchsize}, learning rate: ${p_lr}, n_epoch: ${n_epoch}," \
                 "n_layers: ${p_depth}, n_kernels: ${n_kernels} \e[0m"

            # HSIC-training (add verbose=1 to print the network architecture)
            uv run hsicbt experiment=resconv-hsicbt model_file=hsic_weight_resconv_${t_data}.pt \
                training_type=hsictrain epochs=${n_epoch} sigma=${p_s} lambda_y=${p_l} \
                learning_rate=${p_lr} data_code=${t_data} batch_size=${p_batchsize} \
                n_layers=${p_depth} hidden_width=${n_kernels}

            # Evaluate with one epoch of format-training
            uv run hsicbt experiment=resconv-format model_file=hsic_weight_resconv_${t_data}.pt \
                training_type=format epochs=1 sigma=${p_s} lambda_y=${p_l} learning_rate=0.1 \
                data_code=${t_data} batch_size=${p_batchsize} n_layers=${p_depth} hidden_width=${n_kernels}
        done
    done
done
