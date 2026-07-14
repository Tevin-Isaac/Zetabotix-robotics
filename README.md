# Zetabotix Robotics

Zetabotix is a robotics software company. We are not building a robot, a robot OS, or a
foundation model — those layers are already well funded and heavily contested. We're
building the tooling layer that every robot learning team needs and nobody has built well
yet.

## First project: pre-training data diagnostics (name TBD)

Robotics teams train policies (ACT, Diffusion Policy, VLAs, ...) on teleoperation datasets,
then discover — hours later, after a full training run — that the data itself was the
problem: a joint that was never moved, an action outside kinematic limits, a dataset too
small for the model architecture. This repo is a CPU-only linter that catches those
problems in the dataset itself, before a single GPU-hour is spent training.

Input: a [LeRobot](https://github.com/huggingface/lerobot)-format dataset.
Output: a report — joint coverage, action feasibility, per-task success ratio, and
model-specific data-sufficiency checks — in under a minute.

This is early. The API, the name, and the report format are all still being shaped by real
datasets. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for where this is headed.

## Status

Pre-alpha. Scaffolding only — no working analyzer yet.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).
