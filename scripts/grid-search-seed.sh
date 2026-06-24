#!/bin/bash

# Example: run the same experiment with different random seeds.
# Uncomment a block below as needed.
for s in {1..10}; do
    echo "Task:${s}, Seed:${RANDOM}"

    # needle case
    # uv run hsicbt experiment=needle training_type=hsictrain epochs=5 seed=${RANDOM}
    # uv run hsicbt-plot task=needle data_code=mnist ext=pdf training_type=hsictrain footnote="random seed: $RANDOM"

    # unformat (hsic-solve) training
    # uv run hsicbt experiment=hsicsolve training_type=hsictrain data_code=mnist seed=${RANDOM}
    # uv run hsicbt-plot task=hsic-solve data_code=mnist ext=pdf footnote="random seed: $RANDOM"
done
