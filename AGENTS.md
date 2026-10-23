# AGENTS.md

Personal research project measuring whole-server energy use of vLLM inference.
`v1/` holds a throwaway first attempt — reference only, not a model to follow.

## Intent

**`docs/VISION.md` is the source of intent.** Read it before doing anything. It
states what we are learning, the configuration, the method decisions already
settled, and what we are explicitly not doing.

The rest of `docs/` is background: the grant this serves, the routing research
program it sits in, and a generic telemetry schema written before the scope was
set. Do not treat those as requirements.

## Shared machine

This is a live server. Other work runs on this box and must keep running.

- The GPU is ours only via the vLLM container already serving Qwen. Do not
  start, stop, reconfigure, or `nvidia-smi -pl` anything, and do not touch
  other containers, processes, or services.
- Experiments are read-only observers plus our own benchmark client. No system
  changes.
- Never leave stray processes behind. Kill everything you start.

## Rules

- Do not commit. The author commits.
- Build only what is asked. No speculative abstraction, config, or features.
- This code is not production software. Handle the normal path; don't write
  code for failures that require several unlikely things to go wrong at once.
- Brief and readable over clever. Plain functions, obvious names, no layers.
- Docstrings are Google style with type hints: an `Args:` section per
  parameter and a `Returns:`/`Yields:` section. Keep them to what the
  function does — rationale and hardware quirks belong in a comment, not the
  docstring.
- Module docstrings state the purpose in a line or two. No rationale.
- Tests only to guard real behavior. Skip them when there is nothing worth
  guarding. Parsing a format we do not control is worth guarding.
