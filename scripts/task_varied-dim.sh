#!/bin/bash
uv run hsicbt experiment=varied-dim hidden_width=8  exp_index=1 training_type=hsictrain
uv run hsicbt experiment=varied-dim hidden_width=32 exp_index=2 training_type=hsictrain
uv run hsicbt experiment=varied-dim hidden_width=64 exp_index=3 training_type=hsictrain
uv run hsicbt experiment=varied-dim hidden_width=8  exp_index=1 epochs=5 training_type=format learning_rate=0.01
uv run hsicbt experiment=varied-dim hidden_width=32 exp_index=2 epochs=5 training_type=format learning_rate=0.01
uv run hsicbt experiment=varied-dim hidden_width=64 exp_index=3 epochs=5 training_type=format learning_rate=0.01
uv run hsicbt-plot task=varied-dim data_code=mnist ext=pdf
