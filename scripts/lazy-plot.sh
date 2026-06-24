#!/bin/bash

# Produce all the result figures at once from existing training logs.
ftype=pdf

# debug training (hsic-train, backprop and format train)
uv run hsicbt-plot task=general ext=$ftype data_code=mnist

# fig2a-c: hsic monitoring in backprop training with varied activation functions
uv run hsicbt-plot task=varied-activation ext=$ftype data_code=mnist

# fig2d-f: hsic monitoring in backprop training with varied network depth
uv run hsicbt-plot task=varied-depth ext=$ftype data_code=mnist

# fig4
uv run hsicbt-plot task=hsic-solve ext=$ftype data_code=mnist
uv run hsicbt-plot task=hsic-solve ext=$ftype data_code=fmnist
uv run hsicbt-plot task=hsic-solve ext=$ftype data_code=cifar10

# fig5: format-training with different hsic-trained networks
uv run hsicbt-plot task=varied-epoch ext=$ftype data_code=mnist

# fig6-a: hsic-trained network capacity with network size
uv run hsicbt-plot task=varied-dim ext=$ftype data_code=mnist

# fig6-b: hsic-trained network capacity with sigma combo
uv run hsicbt-plot task=sigma-combined ext=$ftype data_code=mnist

# fig7: resnet case
uv run hsicbt-plot task=resconv ext=$ftype
