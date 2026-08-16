"""CLI for the development-only A2 optimization-budget trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner_toy.a2_optimization_budget_trajectory import run, validate_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_trajectory(
            args.output_dir,
            implementation_commit=args.implementation_commit,
        )
        print(result)
        return
    result = run(
        args.output_dir,
        implementation_commit=args.implementation_commit,
    )
    print(
        {
            "canonical_identity": result["canonical_identity"],
            "source_sha256": result["source_sha256"],
            "checkpoint_epochs": result["checkpoint_epochs"],
        }
    )


if __name__ == "__main__":
    main()
