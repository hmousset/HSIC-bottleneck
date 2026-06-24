#!/bin/bash

# Quick 1-epoch smoke run of every experiment configuration.
uv run hsicbt experiment=general-backprop   epochs=1
uv run hsicbt experiment=general-hsicbt      epochs=1
uv run hsicbt experiment=general-format      epochs=1
uv run hsicbt experiment=hsicsolve           epochs=1
uv run hsicbt experiment=needle              epochs=1
uv run hsicbt experiment=resconv-backprop    epochs=1
uv run hsicbt experiment=resconv-format      epochs=1
uv run hsicbt experiment=resconv-hsicbt      epochs=1
uv run hsicbt experiment=sigma-combined      epochs=1
uv run hsicbt experiment=varied-activation   epochs=1
uv run hsicbt experiment=varied-depth        epochs=1
uv run hsicbt experiment=varied-dim          epochs=1
uv run hsicbt experiment=varied-epoch        epochs=1
