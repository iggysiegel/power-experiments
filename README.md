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

```
uv run scripts/extract.py experiments/qwen3.8-27b.yml    # record every run
uv run scripts/transform.py                              # tabulate them
uv run scripts/plot.py                                   # fit and plot them
```

Each run records into `results/<run>-<utc time>/`, the table goes to
`results/results.csv`, and the figures to `results/*.png`.
