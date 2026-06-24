#!/bin/bash
uv run hsicbt experiment=varied-activation exp_index=1 atype=relu    training_type=backprop
uv run hsicbt experiment=varied-activation exp_index=2 atype=tanh    training_type=backprop
uv run hsicbt experiment=varied-activation exp_index=3 atype=elu     training_type=backprop
uv run hsicbt experiment=varied-activation exp_index=4 atype=sigmoid training_type=backprop
uv run hsicbt-plot task=varied-activation data_code=mnist ext=pdf
