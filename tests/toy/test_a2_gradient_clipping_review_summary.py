from __future__ import annotations

import copy
from pathlib import Path

import pytest

from planner_toy.a2_gradient_clipping_review_summary import (
    MAX_REVIEW_SUMMARY_BYTES,
    REVIEW_SUMMARY_FILENAME,
    build_review_summary,
    validate_review_summary,
    write_review_summary,
)


def _payload() -> dict:
    seeds = [17, 29, 43]
    arms = {
        "clip_1_0": {"clip_threshold": 1.0},
        "clip_5_0": {"clip_threshold": 5.0},
        "no_clip": {"clip_threshold": None},
    }
    results = []
    clipping = []
    commitments = []
    for arm, arm_meta in arms.items():
        for seed in seeds:
            rescue_events = {
                "first_position0_operator_rescue": {"epoch": 10, "update_count": 30},
                "first_full_free_running_rescue": {"epoch": 20, "update_count": 60},
            }
            persistence = {
                "position0_operator_rescue": {"10": True, "30": True, "100": True},
                "full_free_running_rescue": {"10": False, "30": True, "100": True},
            }
            results.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "clip_threshold": arm_meta["clip_threshold"],
                    "rescue_events": rescue_events,
                    "rescue_persistence": persistence,
                    "initialization_canonical_sha256": f"sha256:init-{seed}",
                    "final_trained_canonical_sha256": f"sha256:model-{arm}-{seed}",
                    "final_optimizer_canonical_sha256": f"sha256:opt-{arm}-{seed}",
                    "gradient_parameter_manifest_sha256": "sha256:manifest",
                }
            )
            clipping.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "first_9_updates": {"update_count": 9, "clipped_update_count": 1},
                    "through_first_position0_rescue_observation": {
                        "event": rescue_events["first_position0_operator_rescue"]
                    },
                    "through_first_full_free_running_rescue_observation": {
                        "event": rescue_events["first_full_free_running_rescue"]
                    },
                    "full_trajectory": {"update_count": 300, "clipped_update_count": 12},
                }
            )
            commitments.append(
                {
                    "version": "a2-gradient-evidence-commitment/1.1",
                    "gradient_hash_version": "a2-named-gradients-exact/1.1",
                    "gradient_parameter_manifest_sha256": "sha256:manifest",
                    "arm": arm,
                    "seed": seed,
                    "update_count": 300,
                    "sha256": f"sha256:grad-{arm}-{seed}",
                }
            )
    control_by_seed = {}
    for seed in seeds:
        control_by_seed[str(seed)] = {
            "seed": seed,
            "status": "PASS",
            "scope": "WHOLE_300_UPDATE_TRAJECTORY",
            "reference_projection_sha256": f"sha256:ref-{seed}",
            "candidate_projection_sha256": f"sha256:ref-{seed}",
            "prefix_9_update_projection_sha256": f"sha256:prefix-{seed}",
            "reference_final_trained_canonical_sha256": f"sha256:model-clip_1_0-{seed}",
            "candidate_final_trained_canonical_sha256": f"sha256:model-clip_1_0-{seed}",
            "reference_final_optimizer_canonical_sha256": f"sha256:opt-clip_1_0-{seed}",
            "candidate_final_optimizer_canonical_sha256": f"sha256:opt-clip_1_0-{seed}",
            "reference_rescue_events": results[0]["rescue_events"],
            "candidate_rescue_events": results[0]["rescue_events"],
            "reference_rescue_persistence": results[0]["rescue_persistence"],
            "candidate_rescue_persistence": results[0]["rescue_persistence"],
            "reference_projection": {"huge": ["never copied to summary"] * 5000},
        }
    return {
        "experiment_version": "development-a2-gradient-clipping/0.1",
        "implementation_commit": "e" * 40,
        "source_sha256": "sha256:source",
        "canonical_identity": "sha256:canonical",
        "heldout_accessed": False,
        "go_latent": "NOT EVALUATED",
        "arms": arms,
        "seeds": seeds,
        "arm_seed_results": results,
        "clipping_summaries": clipping,
        "gradient_evidence_commitments": commitments,
        "control_equivalence": {
            "reference": "accepted canonical sufficient-budget A2 canonical_order path",
            "required_status": "PASS",
            "by_seed": control_by_seed,
        },
        "cross_seed_clipping_aggregates": {
            arm: {"seed_count": 3} for arm in arms
        },
        "paired_causal_contrasts": {
            "clip_5_0": {"by_seed": {}},
            "no_clip": {"by_seed": {}},
        },
        "intervention_consistency": {
            str(seed): {"first_actual_or_pregradient_difference_update_index": 0}
            for seed in seeds
        },
        "interpretation_policy": {
            "reviewer_owns_interpretation": True,
            "p_value_testing": None,
        },
    }


def test_summary_is_compact_and_excludes_large_reference_projection(
    tmp_path: Path,
) -> None:
    payload = _payload()
    summary = build_review_summary(payload)
    assert "reference_projection" not in summary["control_equivalence"]["by_seed"]["17"]
    producer_output = tmp_path / "producer-evidence"
    producer_output.mkdir()
    path = write_review_summary(producer_output, payload)
    assert path.name == REVIEW_SUMMARY_FILENAME
    assert path.stat().st_size < MAX_REVIEW_SUMMARY_BYTES
    assert validate_review_summary(producer_output, payload) == summary


def test_summary_tamper_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    producer_output = tmp_path / "producer-evidence"
    producer_output.mkdir()
    path = write_review_summary(producer_output, payload)
    path.write_text(
        path.read_text(encoding="utf-8").replace("NOT EVALUATED", "EVALUATED"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="A2_CLIP_REVIEW_SUMMARY_MISMATCH"):
        validate_review_summary(producer_output, payload)


def test_summary_changes_when_claim_bearing_payload_changes(tmp_path: Path) -> None:
    payload = _payload()
    producer_output = tmp_path / "producer-evidence"
    producer_output.mkdir()
    write_review_summary(producer_output, payload)
    changed = copy.deepcopy(payload)
    changed["paired_causal_contrasts"]["no_clip"] = {
        "by_seed": {"17": {"delta": 3}}
    }
    with pytest.raises(ValueError, match="A2_CLIP_REVIEW_SUMMARY_MISMATCH"):
        validate_review_summary(producer_output, changed)
