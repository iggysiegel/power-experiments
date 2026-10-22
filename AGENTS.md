# AGENTS.md

Personal research project measuring whole-server energy use of vLLM inference.
`v1/` holds a throwaway first attempt — reference only, not a model to follow.

## Intent

**`VISION.md` is the source of intent.** Read it before doing anything. It
states what we are learning, the configuration, the method decisions already
settled, and what we are explicitly not doing.

`docs/` is background: the grant this serves, the routing research program it
sits in, and a generic telemetry schema written before the scope was set.
Do not treat `docs/` as requirements.

## Rules

- Do not commit. The author commits.
- Build only what is asked. No speculative abstraction, config, or features.
- This is not production. Handle the normal path; don't write code for
  failures that require several unlikely things to go wrong at once.
- Brief and readable over clever. Plain functions, obvious names, no layers.
- Tests only to guard real behavior. Skip them when there is nothing worth
  guarding.

## Research questions

Settled. See `VISION.md`. Do not infer them from the v1 code.
