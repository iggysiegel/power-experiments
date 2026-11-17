# AGENTS.md

Research project measuring the energy use of vLLM inference on one server.

Run everything through `uv`: `uv run pytest`, `uv run ruff check`,
`uv run ruff format`. `README.md` has the experiment pipeline.

## Shared machine

Other work runs on this server and must keep running.

- The only thing you may change is whether a vLLM container named by
  `container:` in `experiments/*.yml` is running: `docker stop` and
  `docker start` it, and leave it running or stopped as you found it. Leave
  its configuration, the GPU settings, and every other container, process,
  and service untouched.
- Kill every other process you start before you finish.

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
