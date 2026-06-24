#!/bin/bash
uv run hsicbt experiment=general-hsicbt   training_type=hsictrain
uv run hsicbt experiment=general-format   training_type=format
uv run hsicbt experiment=general-backprop training_type=backprop
uv run hsicbt-plot task=general data_code=mnist ext=pdf
