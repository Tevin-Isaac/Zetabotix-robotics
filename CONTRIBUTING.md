# Contributing

This project is early — the core analyzer doesn't exist yet, so the most valuable
contributions right now are:

- Trying the analyzer against real LeRobot datasets once it lands, and reporting what it
  gets wrong
- Domain knowledge: known failure signatures in robot training data, sensible thresholds
  for coverage/feasibility checks, per-policy-type data requirements
- Small, focused PRs — this is not the place for large speculative rewrites yet

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Pull requests

- Keep PRs scoped to one change
- Add or update a test for any behavior change
- Explain the *why* in the PR description, not just the *what*
