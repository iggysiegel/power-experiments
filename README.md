# power-experiments

Measures the power and energy of serving LLM inference with vLLM on one
server. See `docs/VISION.md` for what we are learning and how.

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
```

Each run records into `results/<run>-<utc time>/`, and the table goes to
`results/results.csv`.
