from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from planner_toy.a2_gradient_clipping_review_summary import (
    MAX_REVIEW_SUMMARY_BYTES,
    REVIEW_SUMMARY_FILENAME,
    build_review_summary,
    validate_review_summary,
    write_review_summary,
)
from scripts import build_a2_gradient_clipping_review_summary as offline_extractor


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
        "cross_seed_clipping_aggregates": {arm: {"seed_count": 3} for arm in arms},
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


def _write_source_evidence(producer_output: Path, payload: dict) -> None:
    producer_output.mkdir(parents=True)
    (producer_output / "a2-gradient-clipping.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    (producer_output / "a2-gradient-clipping.md").write_text(
        "derivative fixture\n",
        encoding="utf-8",
    )


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
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


def test_summary_over_64_kib_fails_closed(tmp_path: Path) -> None:
    payload = _payload()
    payload["paired_causal_contrasts"]["no_clip"]["by_seed"]["17"] = {
        "oversized": "x" * MAX_REVIEW_SUMMARY_BYTES
    }
    producer_output = tmp_path / "producer-evidence"
    producer_output.mkdir()
    summary_path = tmp_path / REVIEW_SUMMARY_FILENAME

    with pytest.raises(ValueError, match="A2_CLIP_REVIEW_SUMMARY_TOO_LARGE"):
        write_review_summary(producer_output, payload)
    assert not summary_path.exists()


def test_offline_extractor_rejects_unvalidated_tampered_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload()
    payload["go_latent"] = "EVALUATED"
    sealed_root = tmp_path / "sealed-evidence"
    producer_output = sealed_root / "producer-evidence"
    _write_source_evidence(producer_output, payload)
    summary_output = tmp_path / "derived" / REVIEW_SUMMARY_FILENAME
    summary_output.parent.mkdir()

    def reject_validation(output: Path, *, implementation_commit: str) -> dict:
        assert output == producer_output
        assert implementation_commit == payload["implementation_commit"]
        raise ValueError("A2_CLIP_VALIDATOR_SCIENCE_BOUNDARY")

    monkeypatch.setattr(offline_extractor, "validate_experiment", reject_validation)
    with pytest.raises(ValueError, match="A2_CLIP_VALIDATOR_SCIENCE_BOUNDARY"):
        offline_extractor.build_offline_review_summary(
            producer_output,
            implementation_commit=payload["implementation_commit"],
            summary_output=summary_output,
        )
    assert not summary_output.exists()


def test_offline_extractor_leaves_source_evidence_tree_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload()
    sealed_root = tmp_path / "sealed-evidence"
    producer_output = sealed_root / "producer-evidence"
    _write_source_evidence(producer_output, payload)
    (sealed_root / "SEALED").write_text("immutable\n", encoding="utf-8")
    before = _tree_snapshot(sealed_root)
    summary_output = tmp_path / "derived" / REVIEW_SUMMARY_FILENAME
    summary_output.parent.mkdir()

    def accept_validation(output: Path, *, implementation_commit: str) -> dict:
        assert output == producer_output
        assert implementation_commit == payload["implementation_commit"]
        return {"valid": True, "source_sha256": payload["source_sha256"]}

    monkeypatch.setattr(offline_extractor, "validate_experiment", accept_validation)
    summary_path, summary, validation = offline_extractor.build_offline_review_summary(
        producer_output,
        implementation_commit=payload["implementation_commit"],
        summary_output=summary_output,
    )

    assert summary_path == summary_output.resolve()
    assert summary["implementation_commit"] == payload["implementation_commit"]
    assert validation["source_sha256"] == payload["source_sha256"]
    assert _tree_snapshot(sealed_root) == before
    assert summary_output.is_file()


def test_offline_extractor_rejects_output_inside_sealed_tree(tmp_path: Path) -> None:
    payload = _payload()
    sealed_root = tmp_path / "sealed-evidence"
    producer_output = sealed_root / "producer-evidence"
    _write_source_evidence(producer_output, payload)
    summary_output = sealed_root / REVIEW_SUMMARY_FILENAME

    with pytest.raises(
        ValueError,
        match="A2_CLIP_REVIEW_SUMMARY_OUTPUT_INSIDE_SEALED_EVIDENCE_ROOT",
    ):
        offline_extractor.build_offline_review_summary(
            producer_output,
            implementation_commit=payload["implementation_commit"],
            summary_output=summary_output,
        )
