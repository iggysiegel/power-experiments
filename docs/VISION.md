# Vision

How much power does one edge server draw to serve LLM inference, and what
shape does the energy take as a function of the workload?

Everything in `docs/` is background. This file is the intent for `v2/`, and
a working document: it changes when we find something better.

## What we are learning

Serving energy is not a single number and not a per-token constant. It splits
into levels, which are watts, and energies, which are joules. They never add
together: quoting a total mixes a fact about the machine being on into a fact
about the work.

| Measurement | Unit | Fitted form |
|---|---|---|
| Off | W | container down, machine on |
| Idle | W | model loaded, no requests |
| Load | J and s, off → answering | one-time |
| Prefill workload | J per request vs. input length *L* | `a + b·L + c·L²` |
| Decode workload | J per output token vs. concurrency *n* | `a + b/n` |
| Agentic workload | J per turn vs. shared prefix length *p* | `a + b·p` |

The levels and load together answer a question of their own: how long the
model must sit unused before stopping the container saves energy.

```
  break-even idle time  =  load above off (J) / (idle (W) − off (W))
```

The workloads are reported as curves with fits rather than a table of
scalars. Curves answer questions at points we did not measure, and their
coefficients should transfer: rerun on a different Qwen and the concurrency
coefficient should move while the per-token floor should not.

## Why this is not already known

Energy is modelled everywhere — including in query routing — as a linear
per-token constant. v1 measured two points against that: batching is worth
~12x per output token (7.1 J at concurrency 1 down to 0.597 J at 16), and a
4096-token prefill took only ~14% more energy than a 128-token one. What was
missing is the *shape* between and around them, which is what makes the
result usable rather than quotable. v2 measures that shape, and does not
reproduce v1's prefill claim (see Results).

## Configuration (fixed for v2)

Recorded so results are reproducible. These are settings, not constants of the
hardware.

- One NVIDIA H100 NVL, 95 GB.
- **Power cap 400 W** (`nvidia-smi -pl`; range 200–400, factory default 310).
  Held fixed for v2; sweeping it is follow-on.
- vLLM serving Qwen/Qwen3.8-27B at 127.0.0.1:8000 in the `vllm-qwen`
  container, text only. Prefix caching is on, the vLLM default. Prefill and
  decode prompts share no prefix, so it does nothing for them; agentic relies
  on it.
- Load generator: `vllm-bench`, random dataset. No external corpus.

## One run shape

Every run is the same sequence with two optional parts:

```
  [ restart: settle → off → load ]  →  settle → idle  →  [ bench: workload ]
            optional                        always            optional
```

- **Settle** waits for the GPU to cool to rest before each resting phase.
- **Restart** stops the container, holds it down (off), then starts it and
  waits for the endpoint to answer (load).
- **Idle** holds the server up with no requests. Every run has one.
- **Bench** runs `vllm-bench` with the run's settings (workload).

A **load run** is restart with no bench. A **workload run** is bench with no
restart, one per curve point, against the server already up.

```
 Load run
   ▲                            ╭─╮ ╭╮
   │                 ╭────────╮ │ ╰─╯╰─╮
   │                 │        ╰─╯      ╰───────────────────
   │ ────────────────╯
   └──┬──────────────┬─────────────────┬──────────────────┬─▶
   stopped         start             ready              end
      │     off      │      load       │      idle        │

 Workload run
   ▲                  ╭────────────────────────────╮
   │                  │░░░░ energy above idle ░░░░░│
   │ ─────────────────╯ - - - - - - - - - - - - - -╰───
   └──┬───────────────┬────────────────────────────┬──▶
    begin       bench starts                   bench ends
      │     idle      │          workload          │
```

Prefill, decode, and agentic are three sets of bench settings, not three code
paths, and differ only in what the energy is divided by. Each curve varies one
setting across runs and holds the rest:

| Workload | Varied across runs | Held fixed | Divided by |
|---|---|---|---|
| Prefill | input length | 1 output token, concurrency 1 | requests |
| Decode | concurrency | input and output length, `ignore-eos` | output tokens |
| Agentic | shared prefix length | 8 turns of short input and output, concurrency 1 | turns |

The agentic workload is `vllm-bench`'s multi-turn mode. One prefix, standing
in for a system prompt and tool definitions, is shared by every conversation
in a run, and each conversation resends its growing history every turn, so the
prefix cache serves most of each prompt.

## Experiment file

One YAML file per experiment in `experiments/`, named for the model and
container setup it runs against: shared settings, then each
named run with its fixed bench settings and an optional `sweep`. A sweep's
lists are read side by side: point *i* takes the *i*-th value of each, so
decode's prompt count scales with its concurrency.

Bench settings are `vllm-bench` flags without their dashes, passed through
unread, so the benchmark's knobs live in the file, not code. `model` is passed
so `vllm-bench` loads the tokenizer locally instead of spending ~50 s asking
the server.

## Pipeline and output

```
uv run scripts/extract.py experiments/qwen3.8-27b.yml   # record every run
uv run scripts/transform.py                             # tabulate
uv run scripts/plot.py                                  # fit and plot
```

`extract.py` writes one directory per run, `results/<name>-<utc time>/`,
holding only raw information, so every number can be recomputed without going
near the GPU again. All timestamps are wall-clock UTC from the same host, so
phases line up with the traces directly. `results/` is not committed.

```
results/decode-20261026T180200/
  config.yml     the settings this run used, seed included
  phases.csv     when each phase began, then `end`
  gpu.csv        GPU trace, 10 Hz
  system.csv     whole-server trace, 1 Hz
  bench.json     what the workload did, per request too   (workload runs)
  container.log  vLLM's timestamped log of the restart     (restart runs)
```

`transform.py` reads every run that reached `end` into `results/results.csv`:
one row per measured phase, with its seconds, and W, J, and J above its
baseline on both traces. Load is measured above its run's off, and workload
above its run's idle. Workload rows also carry the requests and output tokens
completed, and every bench flag as set.

`plot.py` fits each curve on both traces, writes `prefill.png`, `decode.png`,
`agentic.png`, `load.png`, and `timeline.png` into `results/`, and prints the
coefficients and break-even times. The timeline shows both traces across every
run with its phases shaded, to spot anything odd during the night.

## Method decisions already settled

- **Own phase per measurement.** Each level and each energy comes from a
  phase the runner began and ended itself, never carved out of a long trace
  afterwards.
- **One run, one point.** A failed run loses one point, not the sweep.
- **Runs are shuffled,** so drift over the night does not line up with any
  sweep.
- **A fresh seed per run.** `vllm-bench` seeds with 0 by default, so runs
  would share prompts and hit the prefix cache. Each run is seeded from its
  start time, so caching helps only within a run.
- **Each workload run subtracts its own idle.** The system trace includes
  everything else on the box, whose draw drifts.
- **Every resting phase starts at rest.** A warm GPU idles several watts high
  (92.5 W at 43 °C against ~100 W at 59 °C), so off and idle are each preceded
  by a settle phase that waits until the GPU is at or below `rest_max_c`.
- **Off begins once the GPU has settled after `docker stop`.**
- **Load is `docker start` to `/v1/models` answering,** which happens only
  after `Application startup complete`.
- **Both power layers, GPU primary.** GPU at 10 Hz via NVML; whole system at
  ~1 Hz via `ipmitool -I open dcmi power reading`, whose integration floor
  sets the rate. The GPU is ~93 W of a ~295 W idle box, so the system trace
  explains rather than headlines.
- **Energy above idle is the primary workload metric.** Total energy is
  mostly a fact about the machine being on.
- **A failed DCMI poll is skipped, not fatal.** It costs about 0.03% of system
  joules.
- **The workload phase is the bench process, launch to exit,** including the
  tail where the last requests finish with fewer in flight (~4% of a decode
  run).
- **Input lengths are exact** via `prompt-token-ids`. Multi-turn needs the
  chat backend, which sends text through the chat template, so agentic prompts
  are not exact.
- **Prefill runs emit one decode token,** and decode runs carry a fixed
  prefill offset. Neither changes a curve's shape.
- **Fits minimise relative error,** since values span decades. Prefill takes
  a quadratic term: a straight line misses the long end by 11%, and attention
  cost grows with context.
- **No one else is using the server during a run.** Assumed, not checked.

## Boundaries fixed by the machine we run on

- **No reboot.** The `vllm-qwen` container may be stopped and started; the box
  may not. So off is the container-down level, not the boot level.
- **Weights-resident is deferred.** vLLM does not idle with weights loaded but
  not serving.
- **Loads are warm.** A restart reads weights from the page cache, not disk,
  and loads compiled kernels from the `vllm-compile-cache` volume (~1.3 s of
  compilation). Dropping either is a system change or a follow-on.

## Results

First full run: `experiments/qwen3.8-27b.yml`, 23 runs, night of
2026-10-26/27, ~3.2 h, 1.5 h of it settling. Every run completed with no
failed requests.

| | GPU | Whole server |
|---|---|---|
| Off | 64 W | 256 W |
| Idle | 93 W | 295 W |
| Load above off (78 s) | 4.1 kJ | 10.3 kJ |
| Break-even idle time | 140 s | 266 s |
| Prefill J/request, `a + b·L + c·L²` | 5.2 + 0.036·L + 2.4e-7·L² | 5.1 + 0.037·L + 8.8e-7·L² |
| Decode J/token, `a + b/n` | 0.063 + 5.5/n | 0.079 + 6.9/n |
| Agentic J/turn, `a + b·p` | 743 + 0.0016·p | 940 + 0.0017·p |

Shapes, for the report to state:

- **Prefill is not strongly sublinear.** Per input token it falls from 0.079 J
  at 128 to ~0.038 J at 2–4k, then rises to 0.043 J at 32k. This contradicts
  v1's "+14% from 128 to 4096".
- **Decode follows `a + b/n` closely,** 37x from concurrency 1 to 64, and is
  still falling at 64, so the floor `a` is extrapolated.
- **Agentic is nearly flat,** +7% from a 1k to a 32k prefix; each turn is
  mostly its 128 output tokens.

Remaining: the report.

## Not doing

- **Dollars, or any cost modeling.** Energy in joules and watts only.
- **Attributing shared power to individual concurrent queries.** No per-request
  energy counter exists on this hardware and MIG does not partition power.
  Measuring J per token *at* a concurrency level is in scope; splitting one
  batch's joules among its members is not.
- **Query routing or model selection.** One endpoint, one model. At most we
  produce an energy function a router could consume.
- **Multi-GPU, multi-node, or multiple serving instances.**
- **Modality beyond text.**
- **Model-family comparisons** (Qwen vs. Gemma). Confounds family with
  training data, tokenizer, and size in a way one GPU cannot untangle.

## Later, if v2 lands

- **Split load into weight transfer and engine init,** from vLLM's startup log
  timestamps, already recorded in `container.log`, cross-checked against the
  10 Hz trace.
- **Cold loads.** Clear the compile cache before a load run; clearing adds
  ~25 s of compilation.
- **Agentic with the prefix cache on and off.** `reset-prefix-cache` needs
  `VLLM_SERVER_DEV_MODE=1` on the container; either is a second experiment
  file.
- **Arrival rate as an axis** alongside concurrency.
- **Dense vs. MoE and quantization,** where energy stops tracking FLOPs. Each
  is a container change and a new experiment file.
