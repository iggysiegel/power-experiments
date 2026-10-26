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
uv run scripts/run.py config.yml <run>    # load, prefill, or decode
uv run pytest
```

Each run records into `results/<run>-<utc time>/`: the GPU and system power
traces, when each phase began, and the run's settings.

## Phases

```
  [ off → load ]  →  idle  →  [ workload ]
     restart         always       bench
```

| Phase | What is happening | Measures |
|---|---|---|
| off | container stopped | W |
| load | container starting, until it answers | J and s |
| idle | model loaded, no requests | W |
| workload | `vllm-bench` running | J above idle, per request (prefill) or per output token (decode) |

The `load` run restarts the container; `prefill` and `decode` run a bench
against the server already up.
