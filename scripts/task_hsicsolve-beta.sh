#!/bin/bash

# test with 5 and 50 hidden layers
n_layers=5
uv run hsicbt experiment=hsicsolve-beta training_type=backprop  data_code=mnist   learning_rate=0.05  n_layers=${n_layers}
uv run hsicbt experiment=hsicsolve-beta training_type=hsictrain data_code=mnist   learning_rate=0.001 sigma=6 lambda_y=200  n_layers=${n_layers}
uv run hsicbt-plot task=hsic-solve data_code=mnist ext=pdf
uv run hsicbt experiment=hsicsolve-beta training_type=backprop  data_code=fmnist  learning_rate=0.05  n_layers=${n_layers}
uv run hsicbt experiment=hsicsolve-beta training_type=hsictrain data_code=fmnist  learning_rate=0.001 sigma=6 lambda_y=1000 n_layers=${n_layers}
uv run hsicbt-plot task=hsic-solve data_code=fmnist ext=pdf
uv run hsicbt experiment=hsicsolve-beta training_type=backprop  data_code=cifar10 learning_rate=0.05  n_layers=${n_layers}
uv run hsicbt experiment=hsicsolve-beta training_type=hsictrain data_code=cifar10 learning_rate=0.001 sigma=6 lambda_y=2500 n_layers=${n_layers}
uv run hsicbt-plot task=hsic-solve data_code=cifar10 ext=pdf
mv assets/exp/fig4-hsic-solve-mnist-test-acc.pdf    assets/exp/fig4-layer5-hsic-solve-mnist-test-acc.pdf
mv assets/exp/fig4-hsic-solve-fmnist-test-acc.pdf   assets/exp/fig4-layer5-hsic-solve-fmnist-test-acc.pdf
mv assets/exp/fig4-hsic-solve-cifar10-test-acc.pdf  assets/exp/fig4-layer5-hsic-solve-cifar10-test-acc.pdf
mv assets/exp/fig3-hsic-solve-actdist-mnist.pdf     assets/exp/fig3-layer5-hsic-solve-actdist-mnist.pdf
mv assets/exp/fig3-hsic-solve-actdist-fmnist.pdf    assets/exp/fig3-layer5-hsic-solve-actdist-fmnist.pdf
mv assets/exp/fig3-hsic-solve-actdist-cifar10.pdf   assets/exp/fig3-layer5-hsic-solve-actdist-cifar10.pdf

n_layers=50
uv run hsicbt experiment=hsicsolve-beta training_type=backprop  data_code=mnist   learning_rate=0.01  n_layers=${n_layers}
uv run hsicbt experiment=hsicsolve-beta training_type=hsictrain data_code=mnist   learning_rate=0.001 sigma=6 lambda_y=2500 n_layers=${n_layers}
uv run hsicbt-plot task=hsic-solve data_code=mnist ext=pdf
uv run hsicbt experiment=hsicsolve-beta training_type=backprop  data_code=fmnist  learning_rate=0.01  n_layers=${n_layers}
uv run hsicbt experiment=hsicsolve-beta training_type=hsictrain data_code=fmnist  learning_rate=0.001 sigma=6 lambda_y=1000 n_layers=${n_layers}
uv run hsicbt-plot task=hsic-solve data_code=fmnist ext=pdf
uv run hsicbt experiment=hsicsolve-beta training_type=backprop  data_code=cifar10 learning_rate=0.01  n_layers=${n_layers}
uv run hsicbt experiment=hsicsolve-beta training_type=hsictrain data_code=cifar10 learning_rate=0.001 sigma=6 lambda_y=500  n_layers=${n_layers}
uv run hsicbt-plot task=hsic-solve data_code=cifar10 ext=pdf
mv assets/exp/fig4-hsic-solve-mnist-test-acc.pdf    assets/exp/fig4-layer50-hsic-solve-mnist-test-acc.pdf
mv assets/exp/fig4-hsic-solve-fmnist-test-acc.pdf   assets/exp/fig4-layer50-hsic-solve-fmnist-test-acc.pdf
mv assets/exp/fig4-hsic-solve-cifar10-test-acc.pdf  assets/exp/fig4-layer50-hsic-solve-cifar10-test-acc.pdf
mv assets/exp/fig3-hsic-solve-actdist-mnist.pdf     assets/exp/fig3-layer50-hsic-solve-actdist-mnist.pdf
mv assets/exp/fig3-hsic-solve-actdist-fmnist.pdf    assets/exp/fig3-layer50-hsic-solve-actdist-fmnist.pdf
mv assets/exp/fig3-hsic-solve-actdist-cifar10.pdf   assets/exp/fig3-layer50-hsic-solve-actdist-cifar10.pdf
