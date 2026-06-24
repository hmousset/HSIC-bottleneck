#!/bin/bash
uv run hsicbt experiment=sigma-combined sigma=1  exp_index=1 training_type=hsictrain
uv run hsicbt experiment=sigma-combined sigma=5  exp_index=2 training_type=hsictrain
uv run hsicbt experiment=sigma-combined sigma=10 exp_index=3 training_type=hsictrain
uv run hsicbt experiment=sigma-combined exp_index=1 training_type=format learning_rate=0.004
uv run hsicbt experiment=sigma-combined exp_index=2 training_type=format learning_rate=0.004
uv run hsicbt experiment=sigma-combined exp_index=3 training_type=format learning_rate=0.004
uv run hsicbt experiment=sigma-combined 'exp_index=[1,2,3]' training_type=format learning_rate=0.004
uv run hsicbt-plot task=sigma-combined data_code=mnist ext=pdf
