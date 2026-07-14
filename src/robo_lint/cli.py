"""Command-line entry point for robo-lint.

Not implemented yet — this is scaffolding for the Phase 1 analyzer described
in docs/ROADMAP.md.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="robo-lint",
        description="Diagnose a robot learning dataset before you train on it.",
    )
    parser.add_argument("dataset", help="Path or Hugging Face repo id of a LeRobot dataset")
    parser.add_argument(
        "--policy-type",
        default=None,
        help="Target policy type (e.g. act, diffusion, smolvla) for readiness checks",
    )
    args = parser.parse_args()

    print(f"robo-lint: analyzer not implemented yet (dataset={args.dataset!r})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
