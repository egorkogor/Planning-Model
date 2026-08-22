"""Compact, validator-checked reviewer summary for A2 gradient-clipping evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, sha256

REVIEW_SUMMARY_VERSION = "a2-gradient-clipping-review-summary/1.0"
REVIEW_SUMMARY_FILENAME = "a2-gradient-clipping-review-summary.json"
MAX_REVIEW_SUMMARY_BYTES = 64 * 1024


def _key(item: dict[str, Any]) -> tuple[str, int]:
    return str(item["arm"]), int(item["seed"])


def _compact_control_equivalence(payload: dict[str, Any]) -> dict[str, Any]:
    control = payload["control_equivalence"]
    by_seed: dict[str, Any] = {}
    keep = (
        "seed",
        "status",
        "scope",
        "reference_projection_sha256",
        "candidate_projection_sha256",
        "prefix_9_update_projection_sha256",
        "reference_final_trained_canonical_sha256",
        "candidate_final_trained_canonical_sha256",
        "reference_final_optimizer_canonical_sha256",
        "candidate_final_optimizer_canonical_sha256",
        "reference_rescue_events",
        "candidate_rescue_events",
        "reference_rescue_persistence",
        "candidate_rescue_persistence",
    )
    for seed, item in sorted(control["by_seed"].items(), key=lambda pair: int(pair[0])):
        by_seed[str(seed)] = {field: item[field] for field in keep}
    return {
        "reference": control["reference"],
        "required_status": control["required_status"],
        "by_seed": by_seed,
    }


def build_review_summary(payload: dict[str, Any]) -> dict[str, Any]:
    results = {_key(item): item for item in payload["arm_seed_results"]}
    clipping = {_key(item): item for item in payload["clipping_summaries"]}
    commitments = {_key(item): item for item in payload["gradient_evidence_commitments"]}

    arm_seed = []
    for arm in payload["arms"]:
        for seed in payload["seeds"]:
            key = (str(arm), int(seed))
            result = results[key]
            arm_seed.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "clip_threshold": result["clip_threshold"],
                    "rescue_events": result["rescue_events"],
                    "rescue_persistence": result["rescue_persistence"],
                    "initialization_canonical_sha256": result[
                        "initialization_canonical_sha256"
                    ],
                    "final_trained_canonical_sha256": result[
                        "final_trained_canonical_sha256"
                    ],
                    "final_optimizer_canonical_sha256": result[
                        "final_optimizer_canonical_sha256"
                    ],
                    "gradient_parameter_manifest_sha256": result[
                        "gradient_parameter_manifest_sha256"
                    ],
                    "gradient_evidence_commitment": commitments[key],
                    "clipping_summary": clipping[key],
                }
            )

    summary: dict[str, Any] = {
        "version": REVIEW_SUMMARY_VERSION,
        "derivative_only": True,
        "claim_bearing_evidence": "producer-evidence/a2-gradient-clipping.json",
        "experiment_version": payload["experiment_version"],
        "implementation_commit": payload["implementation_commit"],
        "source_sha256": payload["source_sha256"],
        "claim_bearing_canonical_identity": payload["canonical_identity"],
        "heldout_accessed": payload["heldout_accessed"],
        "go_latent": payload["go_latent"],
        "arm_seed": arm_seed,
        "control_equivalence": _compact_control_equivalence(payload),
        "cross_seed_clipping_aggregates": payload["cross_seed_clipping_aggregates"],
        "paired_causal_contrasts": payload["paired_causal_contrasts"],
        "intervention_consistency": payload["intervention_consistency"],
        "interpretation_policy": payload["interpretation_policy"],
        "review_guidance": {
            "default_read_path": REVIEW_SUMMARY_FILENAME,
            "full_evidence_read_policy": (
                "Do not stream the full claim-bearing JSON into an LLM context; materialize "
                "and parse it locally only when a claim requires raw trace inspection."
            ),
        },
    }
    summary["summary_sha256"] = sha256(summary)
    return summary


def _serialized(summary: dict[str, Any]) -> bytes:
    data = canonical_bytes(summary) + b"\n"
    if len(data) > MAX_REVIEW_SUMMARY_BYTES:
        raise ValueError(
            f"A2_CLIP_REVIEW_SUMMARY_TOO_LARGE:{len(data)}:{MAX_REVIEW_SUMMARY_BYTES}"
        )
    return data


def review_summary_path(producer_output: Path) -> Path:
    return producer_output.parent / REVIEW_SUMMARY_FILENAME


def _resolved_summary_path(producer_output: Path, output_path: Path | None) -> Path:
    return output_path if output_path is not None else review_summary_path(producer_output)


def write_review_summary(
    producer_output: Path,
    payload: dict[str, Any],
    *,
    output_path: Path | None = None,
) -> Path:
    path = _resolved_summary_path(producer_output, output_path)
    path.write_bytes(_serialized(build_review_summary(payload)))
    return path


def validate_review_summary(
    producer_output: Path,
    payload: dict[str, Any],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    path = _resolved_summary_path(producer_output, output_path)
    if not path.is_file():
        raise ValueError("A2_CLIP_REVIEW_SUMMARY_MISSING")
    expected = build_review_summary(payload)
    expected_bytes = _serialized(expected)
    actual_bytes = path.read_bytes()
    if len(actual_bytes) > MAX_REVIEW_SUMMARY_BYTES:
        raise ValueError("A2_CLIP_REVIEW_SUMMARY_TOO_LARGE")
    if actual_bytes != expected_bytes:
        raise ValueError("A2_CLIP_REVIEW_SUMMMARY_MISMATCH")
    loaded = json.loads(actual_bytes)
    if loaded != expected:
        raise ValueError("A2_CLIP_REVIEW_SUMMMARY_CANONICALIZATION_MISMATCH")
    return expected
