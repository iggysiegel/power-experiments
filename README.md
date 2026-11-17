# power-experiments

Measures the power and energy of serving LLM inference with vLLM on one
server.

## Setup

```
uv sync
```

Also needs `vllm-bench` on the path, Docker access, and passwordless
`sudo ipmitool`.

## Run

Results are kept per campaign, then per experiment. Record each experiment
file into a named campaign, then tabulate and plot the whole campaign:

```
uv run scripts/extract.py 2026-11-final experiments/qwen3.8-27b.yml  # record
uv run scripts/transform.py results/2026-11-final                    # tabulate
uv run scripts/plot.py results/2026-11-final                         # plot
```

Each run records into `results/<campaign>/<experiment>/<run>-<utc time>/`,
where the experiment is the experiment file's name. The campaign's table goes
to `results/<campaign>/results.csv`, and its figures to
`results/<campaign>/*.png`.