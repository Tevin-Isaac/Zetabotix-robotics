# robo-lint

> **Status: reference implementation, not under active standalone development.**
> While validating this against real datasets we found
> [`lerobot-doctor`](https://github.com/jashshah999/lerobot-doctor), a more complete tool
> already covering this ground and already integrated into Hugging Face's official
> [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).
> The one confirmed gap — checking actions against a robot's real kinematic limits (from a
> URDF), not just dataset-internal statistics — is being built as a contribution to that
> project instead of a competing tool here. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for
> the full reasoning. This repo stays up as a working, tested implementation of the
> original checks and a record of how we got here.

**A linter for robot training data.** Point it at a dataset before you train a policy on
it, and it tells you — in seconds, on a CPU — whether the data is actually going to
produce a working policy.

## The problem

Training a robot policy (ACT, Diffusion Policy, a VLA fine-tune, ...) usually takes hours,
often on a GPU you're paying for. A common workflow looks like this:

1. Collect a dataset (teleoperation, sim rollouts, whatever)
2. Train for several hours
3. Deploy to the robot
4. Watch it fail
5. Spend more hours in logs/videos figuring out why
6. Discover the actual cause was in the dataset all along — a joint that was barely ever
   moved during collection, actions that fall outside what the arm can physically reach,
   or simply not enough episodes for the model architecture being used

Steps 2–5 are avoidable. Most of what causes them is visible by just looking at the
dataset, before training starts.

## What robo-lint does

Given a dataset in [LeRobot](https://github.com/huggingface/lerobot) format, robo-lint
runs four checks and prints a report:

| Check | What it catches |
|---|---|
| **Joint coverage** | Joints/axes that barely move across the whole dataset ("dead joints") — the policy learns "do nothing" is always correct for that joint, and only fails when it's actually needed |
| **Action feasibility** | Actions outside the dataset's declared kinematic limits, when those limits are known |
| **Success ratio** | Per-episode success rate, when the dataset records one |
| **Model readiness** | Episode count vs. a heuristic minimum for the target policy type (e.g. VLAs generally need more demonstrations than ACT for the same task) |

Any check that can't be run for a given dataset (e.g. no kinematic limits available) is
reported as **skipped**, not silently passed — the report tells you what it did and didn't
verify.

## Installation

```bash
git clone https://github.com/Tevin-Isaac/Zetabotix-robotics.git
cd Zetabotix-robotics
pip install -e .
```

Requires Python 3.10+.

## Usage

```bash
robo-lint <local-path-or-hf-repo-id> [--policy-type act|diffusion|smolvla|vla] [--json]
```

Example, run against a synthetic dataset where one joint (`wrist_yaw`) was never actually
moved during collection, and there are fewer episodes than an ACT policy typically needs:

```
$ robo-lint /path/to/dataset --policy-type act

# robo-lint report: /path/to/dataset

- Frames: 1200
- Episodes: 40

## Joint coverage
- shoulder_pitch: relative range 0.80
- shoulder_roll: relative range 0.61
- elbow: relative range 1.00
- wrist_yaw: relative range 0.00  ⚠️ DEAD JOINT
- gripper: relative range 0.47

## Action feasibility
- Skipped: no kinematic limits available for this dataset

## Success ratio
- Skipped: no success signal found in this dataset

## Model readiness
- act: 40 episodes (⚠️ UNDER-RESOURCED, heuristic minimum 50)
  note: Heuristic minimum from community-reported training experience, not a hard rule.
```

`robo-lint` exits non-zero when it finds a dead joint or an under-resourced dataset, so it
can be dropped into CI to block a training run before it starts.

## How it works

- `robo_lint/loaders.py` reads a LeRobot dataset's per-frame `action` / `observation.state`
  / `episode_index` columns straight out of its parquet files — this works for both the
  v2.1 (one file per episode) and v3.0 (many episodes per file) dataset layouts, since both
  store the same columns. A local path or a Hugging Face `repo_id` both work; the latter is
  downloaded via `huggingface_hub`.
- `robo_lint/checks.py` runs the four checks above against plain numpy arrays — no
  dataset-format knowledge lives here, only the logic.
- `robo_lint/report.py` turns the check results into Markdown or JSON.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for what's implemented vs. planned.

## Project status

Phase 1 (the analyzer above) is implemented and tested. Not yet done: validation against
real public LeRobot Hub datasets, a pip-installable release, and a hosted dashboard. See
the roadmap for the full picture.

## Contributing

The most useful contribution right now is running `robo-lint` against a real dataset and
telling us what it got wrong — the thresholds and heuristics (what counts as a "dead"
joint, minimum episodes per policy type) are educated guesses that need real-world
correction. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup instructions and PR
expectations.

## License

MIT — see [`LICENSE`](LICENSE).
