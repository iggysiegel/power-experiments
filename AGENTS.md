# AGENTS.md

Research project measuring the energy use of vLLM inference on one server.

## Shared machine

Other work runs on this server and must keep running.

- Only stop and start the `vllm-qwen` container. Do not reconfigure it, change
  GPU settings, or touch other containers, processes, or services.
- Kill every process you start.

## Rules

- Do not commit. The author commits.
- Build only what is asked. Simple, readable code over clever code.
- Handle the normal path, not unlikely failures.
- Google-style docstrings with an `Args:` and `Returns:`/`Yields:` section.
  Rationale goes in comments, not docstrings.
- Test only real behavior worth guarding, such as parsing formats we do not
  control.
