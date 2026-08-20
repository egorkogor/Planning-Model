"""CLI for the train-only A2 gradient-clipping causal microexperiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner_toy.a2_gradient_clipping import run, validate_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(
            validate_experiment(
                args.output_dir, implementation_commit=args.implementation_commit
            )
        )
        return
    result = run(args.output_dir, implementation_commit=args.implementation_commit)
    print(
        {
            "canonical_identity": result["canonical_identity"],
            "source_sha256": result["source_sha256"],
            "arms": list(result["arms"]),
            "seeds": result["seeds"],
            "go_latent": result["go_latent"],
        }
    )


if __name__ == "__main__":
    main()
