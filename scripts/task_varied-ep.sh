#!/bin/bash
uv run hsicbt experiment=varied-epoch epochs=1  exp_index=1 training_type=hsictrain
uv run hsicbt experiment=varied-epoch epochs=5  exp_index=2 training_type=hsictrain
uv run hsicbt experiment=varied-epoch epochs=10 exp_index=3 training_type=hsictrain
uv run hsicbt experiment=varied-epoch epochs=5  exp_index=1 training_type=format learning_rate=0.005
uv run hsicbt experiment=varied-epoch epochs=5  exp_index=2 training_type=format learning_rate=0.005
uv run hsicbt experiment=varied-epoch epochs=5  exp_index=3 training_type=format learning_rate=0.005
uv run hsicbt-plot task=varied-epoch data_code=mnist ext=pdf
