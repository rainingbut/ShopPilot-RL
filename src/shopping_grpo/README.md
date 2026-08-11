# Package layout

The package follows the tutorial pipeline:

```text
environment/       connect to ShopSimulator and enforce the action contract
training/sft/      build assistant-only SFT examples
training/grpo/     connect the shopping AgentLoop and reward to veRL
evaluation/        hard checks, Rubric curation, trajectory Judge and aggregation
cli.py             small installed-package commands
smoke.py           CPU-only public smoke path
```

User-facing commands remain in the repository-level `scripts/` directory.
Those launchers call these modules; they are not a second implementation.
