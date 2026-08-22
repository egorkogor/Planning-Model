"""Build a compact reviewer summary from independently validated A2 clipping evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from planner_toy.a2_gradient_clipping import OUTPUT_JSON, validate_experiment
from planner_toy.a2_gradient_clipping_review_summary import (
    REVIEW_SUMMARY_FILENAME,
    validate_review_summary,
    write_review_summary,
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validated_summary_output_path(producer_output_dir: Path, summary_output: Path) -> Path:
    producer_output_dir = producer_output_dir.resolve()
    sealed_evidence_root = producer_output_dir.parent
    summary_output = summary_output.resolve()
    if _is_within(summary_output, sealed_evidence_root):
        raise ValueError("A2_CLIP_REVIEW_SUMMARY_OUTPUT_INSIDE_SEALED_EVIDENCE_ROOT")
    if summary_output.name != REVIEW_SUMMARY_FILENAME:
        raise ValueError("A2_CLIP_REVIEW_SUMMARY_OUTPUT_FILENAME")
    return summary_output


def build_offline_review_summary(
    producer_output_dir: Path,
    *,
    implementation_commit: str,
    summary_output: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    producer_output_dir = producer_output_dir.resolve()
    summary_output = _validated_summary_output_path(producer_output_dir, summary_output)

    # Fail closed before reading fields for summary derivation. This independently
    # validates the sealed producer evidence and binds it to the accepted commit/source.
    validation = validate_experiment(
        producer_output_dir,
        implementation_commit=implementation_commit,
    )

    payload_path = producer_output_dir / OUTPUT_JSON
    with payload_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("implementation_commit") != implementation_commit:
        raise ValueError("A2_CLIP_REVIEW_SUMMARY_IMPLEMENTATION_MISMATCH")
    if validation.get("source_sha256") != payload.get("source_sha256"):
        raise ValueError("A2_CLIP_REVIEW_SUMMARY_VALIDATED_SOURCE_MISMATCH")

    summary_path = write_review_summary(
        producer_output_dir,
        payload,
        output_path=summary_output,
    )
    summary = validate_review_summary(
        producer_output_dir,
        payload,
        output_path=summary_output,
    )
    return summary_path, summary, validation


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-output-dir", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary_path, summary, validation = build_offline_review_summary(
        args.producer_output_dir,
        implementation_commit=args.implementation_commit,
        summary_output=args.summary_output,
    )
    print(
        {
            "review_summary": summary_path.name,
            "review_summary_path": str(summary_path),
            "review_summary_bytes": summary_path.stat().st_size,
            "review_summary_sha256": summary["summary_sha256"],
            "claim_bearing_canonical_identity": summary[
                "claim_bearing_canonical_identity"
            ],
            "implementation_commit": summary["implementation_commit"],
            "source_sha256": validation["source_sha256"],
            "go_latent": summary["go_latent"],
        }
    )


if __name__ == "__main__":
    main()
