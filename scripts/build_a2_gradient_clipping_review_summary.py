"""Build the compact reviewer summary from an existing A2 clipping evidence directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planner_toy.a2_gradient_clipping import OUTPUT_JSON
from planner_toy.a2_gradient_clipping_review_summary import (
    REVIEW_SUMMARY_FILENAME,
    validate_review_summary,
    write_review_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-output-dir", required=True, type=Path)
    args = parser.parse_args()
    payload_path = args.producer_output_dir / OUTPUT_JSON
    with payload_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    summary_path = write_review_summary(args.producer_output_dir, payload)
    summary = validate_review_summary(args.producer_output_dir, payload)
    print(
        {
            "review_summary": REVIEW_SUMMARY_FILENAME,
            "review_summary_path": str(summary_path),
            "review_summary_bytes": summary_path.stat().st_size,
            "review_summary_sha256": summary["summary_sha256"],
            "claim_bearing_canonical_identity": summary[
                "claim_bearing_canonical_identity"
            ],
            "go_latent": summary["go_latent"],
        }
    )


if __name__ == "__main__":
    main()
