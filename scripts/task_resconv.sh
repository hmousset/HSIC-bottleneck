#!/bin/bash
uv run hsicbt experiment=resconv-hsicbt   model_file=hsic_weight_resconv_mnist.pt   data_code=mnist
uv run hsicbt experiment=resconv-format   model_file=hsic_weight_resconv_mnist.pt   data_code=mnist
uv run hsicbt experiment=resconv-backprop model_file=hsic_weight_resconv_mnist.pt   data_code=mnist
uv run hsicbt experiment=resconv-hsicbt   model_file=hsic_weight_resconv_fmnist.pt  data_code=fmnist sigma=5 lambda_y=500
uv run hsicbt experiment=resconv-format   model_file=hsic_weight_resconv_fmnist.pt  data_code=fmnist
uv run hsicbt experiment=resconv-backprop model_file=hsic_weight_resconv_fmnist.pt  data_code=fmnist
uv run hsicbt experiment=resconv-hsicbt   model_file=hsic_weight_resconv_cifar10.pt data_code=cifar10 sigma=5 lambda_y=500
uv run hsicbt experiment=resconv-format   model_file=hsic_weight_resconv_cifar10.pt data_code=cifar10
uv run hsicbt experiment=resconv-backprop model_file=hsic_weight_resconv_cifar10.pt data_code=cifar10
uv run hsicbt-plot task=resconv ext=pdf
