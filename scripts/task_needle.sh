#!/bin/bash
##### deprecated in current paper.
task=hsictrain  # backprop/hsictrain
uv run hsicbt experiment=needle training_type=${task} epochs=50
uv run hsicbt-plot task=needle data_code=mnist ext=pdf training_type=${task}
