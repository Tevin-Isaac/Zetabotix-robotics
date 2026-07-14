"""Command-line entry point for robo-lint."""

import argparse
import json
import sys

from .analyzer import analyze_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="robo-lint",
        description="Diagnose a robot learning dataset before you train on it.",
    )
    parser.add_argument("dataset", help="Local path or Hugging Face repo id of a LeRobot dataset")
    parser.add_argument(
        "--policy-type",
        default=None,
        help="Target policy type (e.g. act, diffusion, smolvla) for readiness checks",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON instead of Markdown")
    args = parser.parse_args()

    try:
        report = analyze_dataset(args.dataset, policy_type=args.policy_type)
    except Exception as exc:
        print(f"robo-lint: failed to analyze {args.dataset!r}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_markdown())

    return 1 if report.has_critical_issues() else 0


if __name__ == "__main__":
    raise SystemExit(main())
