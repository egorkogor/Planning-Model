"""CLI for the train-only A2 gradient-clipping causal microexperiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planner_toy.a2_gradient_clipping import OUTPUT_JSON, run, validate_experiment
from planner_toy.a2_gradient_clipping_review_summary import (
    REVIEW_SUMMARY_FILENAME,
    validate_review_summary,
    write_review_summary,
)


def _load_payload(output_dir: Path) -> dict:
    return json.loads((output_dir / OUTPUT_JSON).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validation = validate_experiment(
            args.output_dir, implementation_commit=args.implementation_commit
        )
        summary = validate_review_summary(args.output_dir, _load_payload(args.output_dir))
        print(
            {
                **validation,
                "review_summary": REVIEW_SUMMARY_FILENAME,
                "review_summary_sha256": summary["summary_sha256"],
            }
        )
        return
    result = run(args.output_dir, implementation_commit=args.implementation_commit)
    summary_path = write_review_summary(args.output_dir, result)
    summary = validate_review_summary(args.output_dir, result)
    print(
        {
            "canonical_identity": result["canonical_identity"],
            "source_sha256": result["source_sha256"],
            "arms": list(result["arms"]),
            "seeds": result["seeds"],
            "review_summary": summary_path.name,
            "review_summary_bytes": summary_path.stat().st_size,
            "review_summary_sha256": summary["summary_sha256"],
            "go_latent": result["go_latent"],
        }
    )


if __name__ == "__main__":
    main()
