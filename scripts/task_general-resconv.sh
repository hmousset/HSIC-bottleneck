#!/bin/bash

####### legacy code #######

# --cifar10/mnist/fashionmnist--
# task_general-resconv.sh [-d] datacode [-e] hsic-training epochs [-t] is hsic-training
#
#   task_general-resconv.sh -d cifar10 -e 50 -t 1
#   task_general-resconv.sh -d mnist   -e 30 -t 1
#   task_general-resconv.sh -d fmnist  -e 50 -t 1

while getopts d:e:t: option; do
    case "${option}" in
        d) datacode=${OPTARG};;
        e) epochs=${OPTARG};;
        t) istrainhsic=${OPTARG};;
    esac
done

echo "${datacode} ${epochs} ${istrainhsic}"

if [ "$istrainhsic" -eq 1 ]; then
    uv run hsicbt experiment=general-hsicbt training_type=hsictrain model=resnet-conv \
        data_code="${datacode}" hidden_width=15 epochs="${epochs}" learning_rate=0.0001 \
        last_hidden_width=512 model_file=hsic_weight_resnetc_"${datacode}".pt verbose=1
fi

uv run hsicbt experiment=general-format   training_type=format   model=resnet-conv \
    data_code="${datacode}" hidden_width=15 epochs=5 learning_rate=0.0025 \
    last_hidden_width=512 model_file=hsic_weight_resnetc_"${datacode}".pt
uv run hsicbt experiment=general-backprop training_type=backprop model=resnet-conv \
    data_code="${datacode}" hidden_width=15 epochs=5 learning_rate=0.0025 last_hidden_width=512
uv run hsicbt-plot task=general data_code="${datacode}" ext=pdf
