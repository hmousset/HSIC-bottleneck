# bin

The standalone `run_hsicbt` / `run_plot` launchers have been replaced by two
console scripts installed by `uv sync` (defined in `pyproject.toml`):

### `hsicbt`
Runs a training / experiment task. Pick an experiment config and override any
key on the command line (Hydra syntax):
```sh
uv run hsicbt experiment=general-hsicbt
# e.g. changing epochs and batch size
uv run hsicbt experiment=general-hsicbt epochs=5 batch_size=32
```
The available experiments live in `src/hsicbt/conf/experiment/`, and the full
list of overridable keys is in `src/hsicbt/config.py`. Logs and models are
written under `assets/`.

### `hsicbt-plot`
Renders the figures for a task:
```sh
uv run hsicbt-plot task=general data_code=mnist ext=pdf
```

See the `scripts/` folder for end-to-end examples.
