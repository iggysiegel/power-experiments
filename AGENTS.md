# AGENTS.md

Research project measuring the energy use of vLLM inference on one server.

Run everything through `uv`: `uv run pytest`, `uv run ruff check`,
`uv run ruff format`. `README.md` has the experiment pipeline.

## Shared machine

This is a live server: other people's services run on it and must keep
running. Its H100 NVL GPU is ours for these experiments. You may:

- Stop any container running on our GPU for a campaign, and start it again
  when the campaign ends, with its configuration unchanged.
- Pull vLLM images.
- Create, start, stop, and remove the experiment containers, the ones named
  by `container:` in `experiments/*.yml`.
- Download those experiments' model weights into the Hugging Face cache, and
  delete only the weights you downloaded.

Leave everything else as you found it: the GPU settings, every other
container, process, and service, and every other cached model. Kill every
process you start before you finish.

## Workflow

- Leave changes uncommitted; the author commits.
- Build only what is asked.

## Docs

- Writing code or tests: [docs/agents/code.md](docs/agents/code.md)

## Agent skills

### Issue tracker

Issues are tracked as local markdown files under `.scratch/<feature>/`. See
[docs/agents/issue-tracker.md](docs/agents/issue-tracker.md).

### Triage labels

Uses the five default triage labels (`needs-triage`, `needs-info`,
`ready-for-agent`, `ready-for-human`, `wontfix`). See
[docs/agents/triage-labels.md](docs/agents/triage-labels.md).

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See
[docs/agents/domain.md](docs/agents/domain.md).
