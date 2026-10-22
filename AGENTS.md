# AGENTS.md

Personal research project measuring whole-server energy use of vLLM inference.
`v1/` holds a throwaway first attempt — reference only, not a model to follow.
All new work goes in `v2/`.

## Rules

- Do not commit. The author commits.
- Build only what is asked. No speculative abstraction, config, or features.
- This is not production. Handle the normal path; don't write code for
  failures that require several unlikely things to go wrong at once.
- Brief and readable over clever. Plain functions, obvious names, no layers.
- Tests only to guard real behavior. Skip them when there is nothing worth
  guarding.

## Research questions

Being pinned down. Do not infer them from the v1 code.
