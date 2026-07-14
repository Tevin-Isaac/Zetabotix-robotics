# Roadmap

## Problem

Robotics teams collect a dataset, train a policy for hours, deploy it, watch it fail, and
only then discover the failure traces back to the data: a dead joint, an action outside
reachable limits, or simply not enough episodes for the model architecture being used. The
data problem is diagnosable before training starts; almost nobody diagnoses it before
training starts.

## Phase 1 — Analyzer MVP

- [ ] Load a LeRobotDataset (v3.0 format: Parquet state/action + MP4 video)
- [ ] Joint coverage: flag joints/axes that stay within a near-zero range across all episodes
- [ ] Action feasibility: flag actions outside declared kinematic limits
- [ ] Per-task success ratio, computed from episode metadata
- [ ] Model-specific readiness: minimum-episode heuristics per policy type (ACT, Diffusion
      Policy, VLA fine-tunes), based on published community findings
- [ ] Report output: Markdown (for GitHub/PRs) and JSON (for programmatic use)

## Phase 2 — Distribution

- [ ] Publish as a pip-installable CLI
- [ ] Hugging Face Space with an interactive dashboard version of the report
- [ ] Validate against 5–10 public LeRobot Hub datasets, publish results
- [ ] Write up findings publicly (what the analyzer actually caught, with before/after)

## Phase 3 — Integration

- [ ] Git pre-training hook / CI check that runs the linter automatically
- [ ] LeRobot ecosystem integration (contribute upstream where it makes sense)
- [ ] Isaac Lab exported-data support

## Explicitly out of scope for now

- Humanoid hardware, robot OS, foundation models, teleoperation data collection as a
  service, blockchain/DePIN — all ruled out; see internal notes for why.
