#!/bin/bash
uv run hsicbt experiment=varied-depth n_layers=5  exp_index=1
uv run hsicbt experiment=varied-depth n_layers=10 exp_index=2
uv run hsicbt experiment=varied-depth n_layers=15 exp_index=3
uv run hsicbt experiment=varied-depth n_layers=20 exp_index=4
uv run hsicbt-plot task=varied-depth data_code=mnist ext=pdf
