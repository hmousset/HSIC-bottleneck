#!/bin/bash

# Datasets are downloaded automatically by torchvision on first use, but this
# script lets you fetch MNIST, FashionMNIST and CIFAR-10 ahead of time into
# assets/data.
uv run python - <<'PY'
from hsicbt.utils.dataset import get_dataset_from_code
for code in ("mnist", "fmnist", "cifar10"):
    print(f"downloading {code} ...")
    get_dataset_from_code(code, batch_size=256)
print("done")
PY
