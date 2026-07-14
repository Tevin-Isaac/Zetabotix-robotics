# Roadmap

## Update (2026-07-14): direction change after validating against the real ecosystem

Phase 1 (below) was built and works — 16 passing tests, and a clean run against the real
`lerobot/pusht` dataset caught a genuine data quirk (its `next.success` column exists but
is never set to `True` for any of its 25,650 frames; robo-lint now flags this as an
unreliable signal instead of reporting a misleading 0% success rate).

But validating against the ecosystem also surfaced
[`lerobot-doctor`](https://github.com/jashshah999/lerobot-doctor): a more mature tool
covering the same ground — dead/stuck actuators, zero-variance features, policy-window
compatibility, per-episode drilldowns — plus `fix`, `trim`, `score`, and `gate`
subcommands we hadn't even scoped. It's already integrated into Hugging Face's official
[LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) as
the "Doctor" tab. Continuing to build a standalone competitor to an already-distributed,
more complete tool isn't a good use of time.

One gap is real and confirmed by reading `lerobot-doctor`'s source directly: **it checks
actions against dataset-internal statistics (empirical min/max), not against a robot's
actual declared kinematic limits (joint position/velocity/torque bounds from a URDF).**
Neither `lerobot-doctor` nor `score_lerobot_episodes` (the other tool found in this space)
does this.

**New plan:** build that one check well, and contribute it to `lerobot-doctor` rather than
maintaining a parallel tool. This repo (`robo_lint`) stays as-is — a working, tested
reference implementation of the original checks, and the record of how this decision was
reached — but active development moves to a fork of `lerobot-doctor`.

## Phase 1 — Analyzer MVP (done, not being extended further here)

- [x] Load a LeRobotDataset (works for both v2.1 and v3.0 layouts)
- [x] Joint coverage: flag joints/axes that stay within a near-zero range across all episodes
- [x] Action feasibility: flag actions outside declared kinematic limits (empirical, from dataset stats)
- [x] Per-episode success ratio, with zero-variance/placeholder-column detection
- [x] Model-specific readiness: minimum-episode heuristics per policy type
- [x] Report output: Markdown and JSON
- [x] Validated against a real public dataset (`lerobot/pusht`)

## Phase 2 — Kinematic limits check (active work, in a fork of lerobot-doctor)

- [ ] Parse joint position/velocity/(torque where declared) limits from a robot's URDF
- [ ] Match dataset action dimensions to URDF joint names (exact match, then best-effort fallback)
- [ ] Flag actions that violate real kinematic limits, distinct from dataset-internal statistical outliers
- [ ] Validate against at least one real dataset + matching URDF (e.g. an SO-101 dataset)
- [ ] Open a PR to `lerobot-doctor` upstream

## Explicitly out of scope for now

- Humanoid hardware, robot OS, foundation models, teleoperation data collection as a
  service, blockchain/DePIN — all ruled out; see internal notes for why.
- Maintaining `robo_lint` as a standalone competing product to `lerobot-doctor` — see the
  update above.
