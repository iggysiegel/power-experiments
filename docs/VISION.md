# Vision

How much energy does it take to serve LLM inference on one edge server, and
what shape does that energy take as a function of the workload?

Everything in `docs/` is background. This file is the intent for `v2/`.

## What we are learning

Serving energy is not a single number and not a per-token constant. It splits
into four stages, each governed by a different binding resource, each with its
own natural unit and its own shape:

| Stage | Unit | Shape |
|---|---|---|
| Baseline | watts (a level, not a cost) | depends on machine state; ~43% of any load window |
| Load | joules, off → ready | one-time integral, split transfer (J/GB) + init (J) |
| Prefill | J per request vs. input length | strongly sublinear |
| Decode | J per output token vs. occupancy | roughly `a + b/occupancy` |

The result is a **cost function per stage**, reported as curves with fits,
rather than a table of scalars. Curves answer questions at points we did not
measure, and their coefficients transfer: rerun on a different Qwen and the
occupancy coefficient should move while the per-token floor should not.

## Why this is not already known

Energy is priced everywhere — including in query routing — as a linear
per-token constant. It is not linear. v1 established the two extremes:
batching is worth ~12x per output token (7.1 J at concurrency 1 down to
0.597 J at 16), and a 4096-token prefill costs only ~14% more than a 128-token
one (~28x cheaper per prefill token). Both are real and both are already
measured. What is missing is the *shape* between and around them, which is what
makes the result usable rather than quotable.

## Configuration (fixed for v2)

Recorded so results are reproducible. These are settings, not constants of the
hardware.

- One NVIDIA H100 NVL, 95 GB.
- **Power cap 400 W** (`nvidia-smi -pl`; range 200–400, factory default 310).
  v1's numbers are all at 400. Hold it fixed for v2; sweeping it is follow-on.
- vLLM serving a current Qwen model at 127.0.0.1:8000, text only.
- Load generator: `vllm-bench`.
- Workload data: synthetic, via `vllm-bench`. No external corpus.

Every setting that varies between runs — model, quantization, lengths,
concurrency, arrival rate, cap, prefix caching — lives in config, not code.
That is the swap mechanism: dense vs. MoE, quantization, a different Qwen, and
the cap sweep are all config changes, not harness changes.

## Workload classes

Two classes, deliberately comparable. Both are expressed on the same axes so
they can share a plot: **arrival rate (req/s) and in-flight concurrency.**

- **Conversational** — short input, moderate output, no shared prefix.
  `--random-prefix-len 0`.
- **Agentic** — the shape of an operational edge workload: a long fixed prefix
  (system prompt + tool schemas), a short instruction, a short output, several
  concurrent sessions, repeated across turns. `--random-prefix-len`,
  `--multi-turn`, `--multi-turn-prefix-global-ratio` (shared tool schemas) and
  `--multi-turn-prefix-conversation-ratio` (per-session growing history).

Prefix caching is a switch, not a caveat. Conversational has no shared prefix
so caching is a no-op there; agentic is all prefix reuse. Run the main surface
with caching off, then one agentic run with it on. The contrast is a result.
`--reset-prefix-cache` (needs `VLLM_SERVER_DEV_MODE=1`) keeps sweep iterations
from contaminating each other.

## Method decisions already settled

- **Own steady window per stage.** Baseline, load, prefill, and decode are
  separate runs with their own start and stop, not phases carved out of one
  long trace. Boundaries stop mattering, and no segmentation step is needed.
- **Both power layers, GPU primary.** GPU at 10 Hz via NVML; whole system at
  ~1 Hz via `ipmitool -I open dcmi power reading` (whose integration floor
  sets the rate). The GPU is ~92 W of a ~294 W idle box, so the system trace
  explains rather than headlines. Record the GPU trace in full because
  transients — KV pool allocation, CUDA graph capture during load — are
  visible only there.
- **Incremental above idle is the primary metric.** Total energy is mostly a
  fact about the machine being on.
- **A failed DCMI poll is skipped, not fatal.** The BMC returns an error every
  few minutes; the gap costs about 0.03% of system joules, so dropping the
  sample and logging it beats discarding the run.
- **One wall-clock anchor per run** joins `--save-detailed` output (relative
  times) to the power trace.
- **Load reported as two terms.** Cold start off → ready is the headline.
  Split into weight transfer (J/GB, transfers across models) and engine init
  (J, against KV pool size: CUDA graph capture, KV allocation). The split comes
  from vLLM's startup log timestamps cross-checked against the 10 Hz trace —
  not from a segmentation algorithm.
- **Prefill runs emit one decode token.** Not worth correcting for.

## Done looks like

Four curves with fits, each from its own steady window, covering both workload
classes where they differ:

- baseline as a small set of levels (fresh boot, weights resident, engine up)
- load as J/GB plus J of init
- prefill: J per request vs. input length
- decode: J per output token vs. occupancy

plus a report stating the shapes and their coefficients, and a harness where
changing the model, quantization, lengths, or cap is a config edit.

## Not doing

- **Dollars, or any cost modeling.** Energy in joules and watts only.
- **Attributing shared power to individual concurrent queries.** No per-request
  energy counter exists on this hardware and MIG does not partition power.
  Measuring J per token *at* an occupancy level is in scope; splitting one
  batch's joules among its members is not.
- **Query routing or model selection.** One endpoint, one model. At most we
  produce a cost function a router could consume.
- **Multi-GPU, multi-node, or multiple serving instances.**
- **Modality beyond text.**
- **Model-family comparisons** (Qwen vs. Gemma). Confounds family with
  training data, tokenizer, and size in a way one GPU cannot untangle.
  Quantization and dense/MoE are the axes that change the cost *shape*, and
  both are reachable as config.

## Later, if v2 lands

- Sweep the power cap (200–400 W). It fans the four stages apart — prefill
  collapses in throughput, load and decode barely move — which is evidence
  that the stages are genuinely different regimes. It also speaks directly to
  the grant's "power-constrained equipment".
- Dense vs. MoE, where energy stops tracking FLOPs and starts tracking expert
  routing and weight bandwidth, making per-token cost input-dependent.
- Multiple serving instances contending for one power budget, as a proxy for
  scheduling across instances.
