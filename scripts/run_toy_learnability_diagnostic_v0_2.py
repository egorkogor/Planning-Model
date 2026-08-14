"""CLI for the extended A2 END-only development learnability diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner_toy.learnability import SEEDS
from planner_toy.learnability_v0_2 import run, validate_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--task-ids", nargs="+")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_diagnostic(args.output_dir)
    else:
        result = run(
            args.output_dir,
            implementation_commit=args.implementation_commit,
            seeds=tuple(args.seeds),
            task_ids=tuple(args.task_ids) if args.task_ids else None,
        )
    print(result)


if __name__ == "__main__":
    main()
