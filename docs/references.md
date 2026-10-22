# References

## Prior work in the program this sits in

Mind the Delay: Latency-aware Query Routing for Heterogeneous LLM pools:
- https://arxiv.org/html/2607.18253v2

Relevant because BEAR/BEA prices cost as a per-token constant and never mentions
energy, power, watts, joules, or kWh. That absence is the gap this project
addresses.

## Datasets

Considered and not used. v2 uses synthetic workloads via `vllm-bench` instead;
see `VISION.md` for the two workload classes.

- lmsys-chat-1m: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
  Gated (needs HF token and accepted terms). Short conversational only — no
  agentic sessions — so it covers one of our two workload classes regardless.
