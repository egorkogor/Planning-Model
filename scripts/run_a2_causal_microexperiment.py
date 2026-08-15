"""CLI for the development-only train-only A2 causal microexperiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner_toy.a2_causal_microexperiment import SEEDS, run, validate_microexperiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    if args.validate_only:
        result = validate_microexperiment(
            args.output_dir,
            implementation_commit=args.implementation_commit,
        )
    else:
        result = run(
            args.output_dir,
            implementation_commit=args.implementation_commit,
            seeds=tuple(args.seeds),
        )
    print(result)


if __name__ == "__main__":
    main()
