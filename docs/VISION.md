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

| Measurement | Unit | Expected shape |
|---|---|---|
| Off | W | container down, machine on |
| Idle | W | model loaded, no requests |
| Load | J and s, off → answering | one-time |
| Prefill workload | J per request vs. input length | strongly sublinear |
| Decode workload | J per output token vs. concurrency | roughly `a + b/concurrency` |
| Agentic workload | J per turn vs. shared prefix length | far flatter than prefill, if the prefix cache works |

The three levels and load together answer a question of their own: how long
the model must sit unused before stopping the container saves energy.

```
  break-even idle time  =  load above off (J) / (idle (W) − off (W))
```

The workloads are reported as curves with fits rather than a table of
scalars. Curves answer questions at points we did not measure, and their
coefficients transfer: rerun on a different Qwen and the concurrency
coefficient should move while the per-token floor should not.

## Why this is not already known

Energy is modelled everywhere — including in query routing — as a linear
per-token constant. It is not linear. v1 established the two extremes:
batching is worth ~12x per output token (7.1 J at concurrency 1 down to
0.597 J at 16), and a 4096-token prefill takes only ~14% more energy than a
128-token one (~28x less per prefill token). Both are real and both are
already measured. What is missing is the *shape* between and around them,
which is what makes the result usable rather than quotable.

## Configuration (fixed for v2)

Recorded so results are reproducible. These are settings, not constants of the
hardware.

- One NVIDIA H100 NVL, 95 GB.
- **Power cap 400 W** (`nvidia-smi -pl`; range 200–400, factory default 310).
  v1's numbers are all at 400. Hold it fixed for v2; sweeping it is follow-on.
- vLLM serving a current Qwen model at 127.0.0.1:8000 in the `vllm-qwen`
  container, text only. Prefix caching is on, the vLLM default; the random
  dataset shares no prefix, so it does nothing for these workloads.
- Load generator: `vllm-bench`, random dataset. No external corpus.

## One run shape

Every run is the same sequence with two optional parts:

```
  [ restart: settle → off → load ]  →  settle → idle  →  [ bench: workload ]
            optional                        always            optional
```

- **Settle** waits for the GPU to cool to rest before each resting phase.
- **Restart** stops the container, holds it down (the off phase), then starts
  it and waits for the endpoint to answer (the load phase).
- **Idle** holds the server up with no requests. Every run has one.
- **Bench** runs `vllm-bench` with the run's settings (the workload phase).

In practice two combinations are used. A **load run** is restart with no
bench: one per repeat of an experiment. A **workload run** is bench with no
restart: many, one per curve point, against the server already up. Combining
both in one run works but is not the normal path — a sweep would mean an
outage and a cold start per point.

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

Prefill, decode, and agentic are not three code paths. They are three sets of
bench settings, and differ only in what the energy is divided by. Each curve
varies one setting across runs and holds the rest:

| Workload | Varied across runs | Held fixed | Divided by |
|---|---|---|---|
| Prefill | input length | 1 output token, concurrency 1 | requests |
| Decode | concurrency | input and output length, `ignore-eos` | output tokens |
| Agentic | shared prefix length | 8 turns of short input and output, concurrency 1 | turns |

The agentic workload is `vllm-bench`'s multi-turn mode: each session shares a
prefix, standing in for a system prompt and tool definitions, and resends its
growing history every turn, so the prefix cache serves most of each prompt.

## Experiment file

One YAML file per experiment in `experiments/`, named for the model and
container setup it runs against. Shared settings sit at the top, then how many
times to repeat the whole experiment, then each named run with its fixed bench
settings and an optional `sweep`. A sweep's lists are read side by side: point
*i* takes the *i*-th value of each, so decode's prompt count scales with its
concurrency.

Bench settings are `vllm-bench` flags written without their dashes and passed
through unread, so the benchmark's knobs live in the file, not code. `model`
is passed to `vllm-bench` so it loads the tokenizer locally; without it, it
spends ~50 s asking the server. An experiment is one invocation:
`uv run scripts/extract.py experiments/qwen3.8-27b.yml`.

```yaml
container: vllm-qwen
url: http://127.0.0.1:8000
model: Qwen/Qwen3.8-27B
idle_s: 30
off_s: 30
rest_max_c: 45
repeats: 3

runs:
  load:
    restart: true

  prefill:
    bench:
      random-output-len: 1
      num-prompts: 100
      max-concurrency: 1
      prompt-token-ids: true
      save-detailed: true
    sweep:
      random-input-len: [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

  decode:
    bench:
      random-input-len: 128
      random-output-len: 512
      ignore-eos: true
      prompt-token-ids: true
      save-detailed: true
    sweep:
      max-concurrency: [1, 2, 4, 8, 16, 32, 64]
      num-prompts: [25, 50, 100, 200, 400, 800, 1600]

  agentic:
    bench:
      backend: openai-chat
      random-input-len: 128
      random-output-len: 128
      num-prompts: 20
      max-concurrency: 1
      multi-turn: true
      multi-turn-num-turns: 8
      ignore-eos: true
      save-detailed: true
    sweep:
      random-prefix-len: [1024, 2048, 4096, 8192, 16384, 32768]
```

## Output

Each run writes one directory, `results/<name>-<utc time>/`, so repeats never
collide. `results/` is not committed.

```
results/decode-20261026T180200/
  config.yml     the settings this run used, seed included
  phases.csv     when each phase began, then `end`
  gpu.csv        GPU trace, 10 Hz
  system.csv     whole-server trace, 1 Hz
  bench.json     what the workload did, per request too   (workload runs)
  container.log  vLLM's timestamped log of the restart     (restart runs)
```

`extract.py` records; `transform.py` computes. A run directory holds only raw
information — what was asked, what happened and when, and the power traces —
so every number is computed afterwards and can be recomputed or replotted
without going near the GPU again. All timestamps are wall-clock UTC from the
same host, so phases line up with the traces directly.

```
timestamp,phase
2026-10-26T22:17:48.746249+00:00,settle
2026-10-26T22:17:48.750952+00:00,off
2026-10-26T22:18:18.751422+00:00,load
2026-10-26T22:19:42.248236+00:00,settle
2026-10-26T22:19:42.257739+00:00,idle
2026-10-26T22:20:12.258109+00:00,end
```

`transform.py` reads every run directory that reached `end` into
`results/results.csv`: one row per measured phase, with its seconds, and W, J,
and J above its baseline on both traces. Load is measured above its run's off,
and workload above its run's idle. Workload rows also carry the requests and
output tokens completed, and every bench flag as set.

## Method decisions already settled

- **Own phase per measurement.** Each level and each energy comes from a
  phase the runner began and ended itself, never carved out of a long trace
  afterwards. No segmentation step is needed.
- **One run, one point.** A workload run produces one curve point; twenty
  lengths means twenty runs. A failed run loses one point, not the sweep.
- **Repeats run the whole experiment, shuffled.** Each repeat records every
  point once in a fresh random order, so drift over the night does not line
  up with any sweep.
- **A fresh seed per run.** `vllm-bench` seeds its random prompts with 0 by
  default, so repeats would send identical prompts and could hit the prefix
  cache. Each run is seeded from its start time instead, so caching helps only
  within a run.
- **Each workload run subtracts its own idle.** The system trace includes
  everything else on the box, whose draw drifts, so an idle level from an
  hour earlier would land its error on every per-token figure. A run's own
  idle phase also stands alone and shows any stray request.
- **Every resting phase starts at rest.** A GPU warm from earlier work idles
  several watts high (92.5 W at 43 °C against ~100 W at 59 °C), and the error
  lands on every figure measured against it. So off and idle are each
  preceded by a settle phase that waits until the GPU is at or below
  `rest_max_c`. Nothing is measured during settle.
- **Off begins once the GPU has settled after `docker stop`.**
- **Load is the trip from `docker start` to the endpoint answering.**
  Readiness is `/v1/models` responding, which happens only after `Application
  startup complete`. Reported as total joules, joules above off, and seconds.
- **Both power layers, GPU primary.** GPU at 10 Hz via NVML; whole system at
  ~1 Hz via `ipmitool -I open dcmi power reading` (whose integration floor
  sets the rate). The GPU is ~92 W of a ~294 W idle box, so the system trace
  explains rather than headlines. Record the GPU trace in full because
  transients — KV pool allocation, CUDA graph capture during load — are
  visible only there.
- **Energy above idle is the primary workload metric.** Total energy is
  mostly a fact about the machine being on.
- **A failed DCMI poll is skipped, not fatal.** The BMC returns an error every
  few minutes; the gap is about 0.03% of system joules, so dropping the
  sample and logging it beats discarding the run.
- **The workload phase is the bench process, launch to exit.** It includes
  the tail where the last requests finish with fewer in flight; at the decode
  settings above that is about 4% of the run, and accepted.
- **Input lengths are exact.** By default `vllm-bench` sends random text that
  the server retokenizes, landing about 5% short of the requested length.
  `prompt-token-ids` sends token IDs instead, so the x-axis is what we set.
  Multi-turn needs the chat backend, which sends text through the chat
  template, so agentic prompts are not exact.
- **Prefill runs emit one decode token.** Not worth correcting for.
- **Decode runs carry a fixed prefill offset.** Every decode request prefills
  its input first, but that length is held constant across a sweep, so the
  offset shifts the curve without changing its shape.
- **No one else is using the server during a run.** Assumed, not checked.

## Boundaries fixed by the machine we run on

- **No reboot.** This is a live server, so the fresh-boot level is out. The
  `vllm-qwen` container may be stopped and started; the box may not.
- **So off is the container-down level, not the boot level.**
- **Weights-resident is deferred.** vLLM does not idle with weights loaded but
  not serving, so the state would need a second copy of the model in GPU
  memory alongside the live container.
- **Weights load from the page cache.** A restart reads the model from host
  RAM, not disk, and dropping the page cache is a system change. Load
  measures that.

## Done looks like

Three levels, load, and three curves with fits:

- off and idle in W, load in J and s, and the break-even idle time
- prefill: J per request vs. input length
- decode: J per output token vs. concurrency
- agentic: J per turn vs. shared prefix length

plus a report stating the shapes and their coefficients.

Widening is more points in a sweep, or another experiment file for another
model or container setup.

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

- **Split load into weight transfer and engine init.** Transfer as J/GB, init
  as J against KV pool size, both from vLLM's startup log timestamps
  (`Model loading took 51.01 GiB memory and 10.22 seconds`, `init engine …
  took 71.91 s (compilation: 24.84 s)`) cross-checked against the 10 Hz trace.
- **Clear the compile cache before a load run.** vLLM caches compiled kernels
  in the `vllm-compile-cache` volume, so the first load runs measure a warm
  start. Clearing adds ~25 s of compilation; worth it if warm and cold differ.
- **Agentic with the prefix cache on and off.** Comparing the two is a
  result; `reset-prefix-cache` needs `VLLM_SERVER_DEV_MODE=1` on the
  container, and either is a second experiment file.
- **Arrival rate as an axis** alongside concurrency.
- Sweep the power cap (200–400 W). It fans the measurements apart — prefill
  collapses in throughput, load and decode barely move — which is evidence
  that they are genuinely different regimes. It also speaks directly to the
  grant's "power-constrained equipment".
- Dense vs. MoE and quantization, where energy stops tracking FLOPs and
  starts tracking expert routing and weight bandwidth. Each is a container
  change, after which the same runs go in a new experiment file.
- Multiple serving instances contending for one power budget, as a proxy for
  scheduling across instances.
