from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / ".research/audits/closed_bottleneck_editability_audit_v1.yaml"
PARENT_AUDIT_PATH = ROOT / ".research/audits/closed_bottleneck_validity_audit_v1.yaml"
PARENT_EVAL_PATH = ROOT / ".research/evals/semantic_feedback_causal_eval_v1.yaml"


def _audit() -> dict:
    return yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8"))


def _parent_audit() -> dict:
    return yaml.safe_load(PARENT_AUDIT_PATH.read_text(encoding="utf-8"))


def _parent_eval() -> dict:
    return yaml.safe_load(PARENT_EVAL_PATH.read_text(encoding="utf-8"))


def _synthetic_status(case: dict) -> str:
    if not case.get("checkpoint_match", True):
        return "INVALID_CHECKPOINT_OR_RETRAINING"
    if case.get("attempted_oracle_full", False):
        return "INVALID_ORACLE_FULL_ATTEMPTED"
    if not case.get("oracle_tier_declared", True):
        return "INVALID_ORACLE_TIER_UNSPECIFIED"
    if not case.get("context_matched", True):
        return "INVALID_CONTEXT_MISMATCH"
    if not case.get("target_declared", True):
        return "INVALID_TARGET_UNDECLARED"
    if not case.get("downstream_position_exists", True):
        return "NOT_EVALUATED_NO_DOWNSTREAM_POSITION"
    if not case.get("edit_applicable", True):
        return "NOT_EVALUATED_EDIT_NOT_APPLICABLE"
    if case.get("directional_scoring_requested", False) and not case.get(
        "matched_pair_available", True
    ):
        return "NOT_EVALUATED_NO_MATCHED_PAIR"
    if case.get("terminated_before_planned_edit", False):
        return "CENSORED_TERMINATED_BEFORE_PLANNED_EDIT"
    return "EVALUATED"


def test_five_way_construct_distinction_is_present_and_not_collapsed() -> None:
    audit = _audit()
    decomposition = audit["construct_decomposition"]
    assert decomposition["collapse_into_single_construct"] == "FORBIDDEN"
    for key in (
        "allowed_channel_usability",
        "normal_trained_model_reliance",
        "targeted_editability",
        "directional_edit_precision",
        "edit_strength_dose_response",
    ):
        assert key in decomposition

    assert decomposition["allowed_channel_usability"]["this_audit_adds"] == (
        "NONE — reference only, no new mechanism"
    )
    assert decomposition["normal_trained_model_reliance"]["this_audit_adds"] == (
        "NONE — reference only, no new mechanism"
    )
    assert decomposition["targeted_editability"]["new_in_this_audit"] is True
    assert decomposition["directional_edit_precision"]["new_in_this_audit"] is True
    assert decomposition["edit_strength_dose_response"]["new_in_this_audit"] is True


def test_estimand_is_defined_and_forbids_collapse_into_reliance_or_usability() -> None:
    audit = _audit()
    estimand = audit["estimand"]
    assert estimand["id"] == "EST-PLN-BOTTLENECK-EDIT-PRECISION"
    assert estimand["rq"] == "RQ-PLN-BOTTLENECK-001"
    assert estimand["status"] == "DESIGN_ONLY_NOT_FROZEN_FOR_EXECUTION"
    forbidden = " ".join(estimand["forbidden_interpretation"])
    assert "normal trained-model reliance" in forbidden
    assert "human-factorized" in forbidden
    assert "pooled single" in forbidden


def test_target_hit_distinct_from_any_change() -> None:
    audit = _audit()
    section = audit["target_hit_vs_any_change"]
    fields = " ".join(section["required_separate_fields"])
    assert "any_change" in fields
    assert "target_hit" in fields
    assert section["collapse_any_change_into_target_hit"] == "FORBIDDEN"


def test_directional_scoring_requires_matched_context_and_declared_target() -> None:
    audit = _audit()
    reqs = audit["directional_scoring_requirements"]
    assert reqs["matched_non_concept_context"]["rule"].startswith("BYTE_IDENTICAL")
    assert reqs["matched_non_concept_context"]["failure"] == "INVALID_CONTEXT_MISMATCH"
    assert reqs["intended_target_declaration"]["failure"] == "INVALID_TARGET_UNDECLARED"
    assert reqs["matched_pairing_for_direction"]["unavailable_state"] == (
        "NOT_EVALUATED_NO_MATCHED_PAIR"
    )


def test_dose_response_requires_more_than_a_single_discrete_swap() -> None:
    audit = _audit()
    grades = audit["edit_strength_and_dose_response_requirements"][
        "dose_response_grade_requirement"
    ]
    assert grades["minimum_predeclared_nonzero_intermediate_grades"] >= 1
    assert grades["minimum_total_grades_including_zero_and_full"] >= 3
    assert grades["insufficient_state"] == "INSUFFICIENT_GRADES_FOR_DOSE_RESPONSE"


def test_oracle_tier_boundary_is_explicit() -> None:
    audit = _audit()
    tiers = audit["oracle_tiers"]
    assert tiers["collapse_into_single_oracle_tier"] == "FORBIDDEN"

    struct_tier = tiers["ORACLE-STRUCT"]
    assert struct_tier["status"] == "SPECIFIABLE_NOW"

    full_tier = tiers["ORACLE-FULL"]
    assert full_tier["status"] == "SPEC_ONLY_NOT_END_TO_END_VALIDATABLE"
    assert "A3s-semantic-target" in full_tier["blocking_dependency"]
    forbidden = full_tier["forbidden_under_this_package"]
    assert "implementing ORACLE-FULL" in forbidden
    assert "executing ORACLE-FULL" in forbidden


def test_interpretation_boundaries_state_usability_is_not_reliance() -> None:
    audit = _audit()
    boundaries = " ".join(audit["interpretation_boundaries"])
    assert "proves channel usability, not normal reliance" in boundaries
    assert "human-factorized semantic coordinate system" in boundaries
    assert "separate SCIENTIFIC Work Package" in boundaries


def test_null_invalidity_and_censoring_states_are_defined() -> None:
    audit = _audit()
    states = audit["null_invalid_and_censoring_states"]["states"]
    expected = {
        "NOT_EVALUATED_NO_DOWNSTREAM_POSITION",
        "NOT_EVALUATED_EDIT_NOT_APPLICABLE",
        "NOT_EVALUATED_NO_MATCHED_PAIR",
        "INVALID_CHECKPOINT_OR_RETRAINING",
        "INVALID_CONTEXT_MISMATCH",
        "INVALID_TARGET_UNDECLARED",
        "INVALID_ORACLE_TIER_UNSPECIFIED",
        "INVALID_ORACLE_FULL_ATTEMPTED",
        "CENSORED_TERMINATED_BEFORE_PLANNED_EDIT",
        "EVALUATED",
    }
    assert expected <= set(states)
    for name, state in states.items():
        if name != "EVALUATED":
            assert state["endpoint"] is None
    assert audit["null_invalid_and_censoring_states"]["null_is_not_zero_effect"] is True


def test_null_state_calibration_matches_declared_precedence() -> None:
    audit = _audit()
    calibration = audit["null_state_calibration"]
    assert calibration["scientific_runtime_invoked"] is False
    assert calibration["outcome_bearing"] is False
    for case in calibration["cases"]:
        assert _synthetic_status(case) == case["expected_status"], case["id"]


def test_evaluated_cases_separate_any_change_from_target_hit() -> None:
    audit = _audit()
    cases = {c["id"]: c for c in audit["null_state_calibration"]["cases"]}

    hit_case = cases["CAL-EDIT-TARGET-HIT"]
    assert hit_case["expected_status"] == "EVALUATED"
    assert hit_case["expected_any_change"] is True
    assert hit_case["expected_target_hit"] is True

    miss_case = cases["CAL-EDIT-ANY-CHANGE-NOT-TARGET"]
    assert miss_case["expected_status"] == "EVALUATED"
    assert miss_case["expected_any_change"] is True
    assert miss_case["expected_target_hit"] is False


def test_future_execution_binding_does_not_authorize_a_new_arm() -> None:
    audit = _audit()
    binding = audit["future_execution_binding"]
    assert binding["new_arm_authorization"] == "NOT_AUTHORIZED_BY_THIS_AUDIT"
    assert binding["execution_work_package"] == (
        "REQUIRES_SEPARATE_EXTERNAL_REVIEWER_AUTHORIZATION_AS_SCIENTIFIC"
    )


def test_promotion_boundary_forbids_scientific_execution_and_preserves_go_latent() -> None:
    audit = _audit()
    boundary = audit["promotion_boundary"]
    assert boundary["scientific_execution"] == "FORBIDDEN"
    assert boundary["edit_intervention_implementation"] == "NOT_AUTHORIZED"
    assert boundary["oracle_full_implementation"] == "NOT_AUTHORIZED"
    assert boundary["go_latent"] == "NOT EVALUATED"


def test_evidence_class_is_non_outcome_bearing_and_read_only_to_frozen_authority() -> None:
    audit = _audit()
    authority = audit["authority"]
    assert authority["evidence_class"] == "DISCOVERY_DESIGN_NON_OUTCOME_BEARING"
    assert authority["frozen_v1_21_read_only"] is True
    non_authority = " ".join(authority["explicit_non_authority"])
    assert "does not modify closed_bottleneck_validity_audit_v1.yaml" in non_authority
    assert "GO_LATENT = NOT EVALUATED" in non_authority


def test_single_writer_surface_is_narrow_and_excludes_parent_files() -> None:
    audit = _audit()
    surfaces = audit["single_writer"]["this_package_writable_surface"]
    assert ".research/audits/closed_bottleneck_editability_audit_v1.yaml" in surfaces
    assert "tests/test_closed_bottleneck_editability_audit.py" in surfaces

    forbidden = audit["single_writer"]["forbidden_writes"]
    assert ".research/audits/closed_bottleneck_validity_audit_v1.yaml" in forbidden
    assert ".research/evals/semantic_feedback_causal_eval_v1.yaml" in forbidden


def test_parent_audit_and_eval_are_referenced_read_only_and_unmodified() -> None:
    audit = _audit()
    refs = audit["read_only_parent_authorities"]
    assert refs["closed_bottleneck_validity_audit"]["path"] == (
        ".research/audits/closed_bottleneck_validity_audit_v1.yaml"
    )
    assert refs["closed_bottleneck_validity_audit"]["access"] == "READ_ONLY_REFERENCE"
    assert refs["semantic_feedback_causal_eval"]["path"] == (
        ".research/evals/semantic_feedback_causal_eval_v1.yaml"
    )
    assert refs["semantic_feedback_causal_eval"]["access"] == "READ_ONLY_REFERENCE"

    # Canary: the accepted parent surfaces must remain unchanged by this package.
    parent_audit = _parent_audit()
    assert parent_audit["version"] == "research-os-bottleneck-audit/0.1"
    assert parent_audit["id"] == "AUD-PLN-CLOSED-BOTTLENECK-001"
    assert "counterfactual_oracle_control" in parent_audit

    parent_eval = _parent_eval()
    assert parent_eval["version"] == "research-os-evaluation-contract/0.1"
    assert parent_eval["id"] == "EVAL-PLN-SEM-FEEDBACK-001"
    estimand_ids = {item["id"] for item in parent_eval["estimands"]}
    assert "EST-PLN-SEM-RELIANCE" in estimand_ids
