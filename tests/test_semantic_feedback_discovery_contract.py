from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / ".research/studies/semantic_feedback_discovery_v1.yaml"


def _protocol() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_three_estimands_remain_distinct() -> None:
    protocol = _protocol()
    estimands = {item["id"]: item for item in protocol["estimands"]}
    assert set(estimands) == {
        "EST-PLN-SEM-ARCH-BUNDLE",
        "EST-PLN-SEM-GEOMETRY",
        "EST-PLN-SEM-RELIANCE",
    }
    assert estimands["EST-PLN-SEM-ARCH-BUNDLE"]["contrast"] == "A3_vs_A2c"
    assert estimands["EST-PLN-SEM-GEOMETRY"]["contrast"] == "A3_vs_A3r"
    assert estimands["EST-PLN-SEM-RELIANCE"]["contrasts"] == [
        "A3_vs_A4",
        "A3_vs_A5",
        "A3_vs_WRONG_SEMANTIC_DONOR",
    ]
    assert protocol["triangulation"]["pooled_semantic_effect"] == "FORBIDDEN"


def test_random_codebooks_are_frozen_before_execution() -> None:
    codebooks = _protocol()["random_codebooks"]
    assert codebooks["count"] == 2
    identities = codebooks["identities"]
    assert [item["id"] for item in identities] == [
        "A3R-CODEBOOK-170029",
        "A3R-CODEBOOK-290043",
    ]
    assert [item["seed"] for item in identities] == [170029, 290043]
    assert len({item["seed"] for item in identities}) == 2
    assert codebooks["selection_after_outcomes"] == "FORBIDDEN"
    assert codebooks["aggregation"]["primary_reporting"] == (
        "REPORT_EACH_CODEBOOK_SEPARATELY"
    )
    assert codebooks["aggregation"]["minimum_defined_codebooks"] == 2


def test_novel_signature_stress_precludes_exact_signature_lookup() -> None:
    stress = _protocol()["novel_compositional_signature_stress"]
    train = set(stress["train_signatures"])
    held = set(stress["stress_signatures"])
    assert train.isdisjoint(held)

    train_atoms = {atom for signature in train for atom in signature.split("|")}
    held_atoms = {atom for signature in held for atom in signature.split("|")}
    assert held_atoms <= train_atoms
    assert stress["contamination_failure"] == (
        "INVALID_GEOMETRY_ESTIMAND_SIGNATURE_OVERLAP"
    )


def test_wrong_semantic_donor_is_deterministic_and_not_relaxable() -> None:
    donor = _protocol()["wrong_semantic_donor"]
    assert donor["checkpoint"] == "EXACT_A3_CHECKPOINT"
    assert donor["retraining"] == "FORBIDDEN"
    assert donor["no_relaxation"] is True
    assert donor["analyst_override"] == "FORBIDDEN"
    assert donor["deterministic_order"] == [
        "ascending absolute difference in feedback_norm_raw",
        "ascending candidate episode_id",
        "ascending candidate unit_id",
    ]
    assert donor["selection"].startswith("first candidate")
    assert donor["donor_unavailable_state"] == "NOT_EVALUATED_DONOR_UNAVAILABLE"


def test_collision_pair_is_metadata_identical_but_semantically_separating() -> None:
    fixture = _protocol()["collision_fixture"]
    first, second = fixture["collision_pair"]
    metadata_fields = fixture["allowed_metadata_fields"]
    for field in metadata_fields:
        assert first[field] == second[field], field
    assert first["semantic_feedback"] != second["semantic_feedback"]
    assert first["required_next_action"] != second["required_next_action"]
    expected = fixture["expected_controls"]
    assert expected["metadata_only_no_feedback"]["collision_pair"] == (
        "INDETERMINATE_MUST_NOT_SEPARATE_TARGETS"
    )
    assert expected["semantic_oracle"]["collision_pair"] == "MUST_SEPARATE_TARGETS"


def test_inverse_metadata_pair_changes_metadata_not_semantics_or_target() -> None:
    fixture = _protocol()["collision_fixture"]
    first, second = fixture["inverse_metadata_pair"]
    assert first["semantic_feedback"] == second["semantic_feedback"]
    assert first["required_next_action"] == second["required_next_action"]
    metadata_fields = fixture["allowed_metadata_fields"]
    assert any(first[field] != second[field] for field in metadata_fields)
    expected = fixture["expected_controls"]
    assert expected["metadata_only_no_feedback"]["inverse_metadata_pair"] == (
        "TARGET_MUST_REMAIN_INVARIANT"
    )
    assert expected["semantic_oracle"]["inverse_metadata_pair"] == (
        "TARGET_MUST_REMAIN_INVARIANT"
    )


def test_same_checkpoint_interventions_are_bound_without_retraining() -> None:
    binding = _protocol()["checkpoint_and_intervention_binding"]
    assert binding["A4_checkpoint"] == "EXACT_A3_CHECKPOINT"
    assert binding["A5_checkpoint"] == "EXACT_A3_CHECKPOINT"
    assert binding["wrong_semantic_donor_checkpoint"] == "EXACT_A3_CHECKPOINT"
    assert binding["A4_A5_donor_retraining"] == "FORBIDDEN"
    assert binding["mismatch_state"] == "INVALID_CHECKPOINT_OR_RETRAINING"


def test_exposure_accounting_is_preintervention_and_endpoints_stay_separate() -> None:
    exposure = _protocol()["exposure_and_endpoints"]
    assert exposure["opportunity_set_source"] == "PRE_INTERVENTION_REFERENCE_PREFIX_ONLY"
    assert exposure["post_treatment_survivor_denominator"] == "FORBIDDEN"
    assert exposure["immediate"]["id"] == "immediate_next_step_effect"
    assert exposure["total"]["id"] == "free_running_total_effect"
    assert exposure["collapse_immediate_and_total"] == "FORBIDDEN"
    assert exposure["treatment_survival_conditioning_for_total"] == "FORBIDDEN"


def test_required_null_and_invalidity_states_are_explicit_and_non_numeric() -> None:
    nulls = _protocol()["null_and_invalidity_states"]
    required = {
        "NOT_EVALUATED_NO_DOWNSTREAM_OPPORTUNITY",
        "NOT_EVALUATED_INTERVENTION_NOT_APPLICABLE",
        "NOT_EVALUATED_DONOR_UNAVAILABLE",
        "INVALID_CHECKPOINT_OR_RETRAINING",
        "INVALID_SHORTCUT_NOT_EXCLUDED",
        "NOT_APPLICABLE_ENDPOINT_UNDEFINED",
        "CENSORED_TERMINATED_BEFORE_PLANNED_INTERVENTION",
    }
    assert required <= set(nulls["states"])
    for state in required:
        assert nulls["states"][state]["endpoint"] is None
    assert nulls["null_is_zero_effect"] is False


def test_collision_and_geometry_failures_are_fail_closed() -> None:
    states = _protocol()["null_and_invalidity_states"]["states"]
    for state in (
        "INVALID_SEMANTIC_CHANNEL_NOT_USABLE",
        "INVALID_METADATA_INVARIANCE",
        "INVALID_GEOMETRY_ESTIMAND_SIGNATURE_OVERLAP",
        "INVALID_GEOMETRY_ESTIMAND_NO_PARTIAL_AGGREGATION",
    ):
        assert states[state]["endpoint"] is None


def test_protocol_remains_preexecution_and_nonconfirmatory() -> None:
    protocol = _protocol()
    governance = protocol["data_governance"]
    boundary = protocol["promotion_boundary"]
    assert protocol["evidence_class"] == "DISCOVERY"
    assert governance["mode"] == "DEVELOPMENT_ONLY_NON_CONFIRMATORY"
    assert governance["legacy_held_out_04_05"] == "FORBIDDEN_NO_ACCESS_NO_MATERIALIZATION"
    assert governance["claim_promotion"] == "FORBIDDEN"
    assert boundary["model_training"] == "FORBIDDEN"
    assert boundary["model_inference"] == "FORBIDDEN"
    assert boundary["scientific_execution"] == "FORBIDDEN"
    assert boundary["execution_work_package"] == (
        "REQUIRES_SEPARATE_EXTERNAL_REVIEWER_AUTHORIZATION"
    )
    assert boundary["go_latent"] == "NOT EVALUATED"


def test_preexecution_calibration_matrix_covers_issue_57_freezes() -> None:
    ids = {item["id"] for item in _protocol()["preexecution_calibration_matrix"]}
    assert ids == {
        "CAL-CODEBOOK-COUNT",
        "CAL-NOVEL-SIGNATURE",
        "CAL-DONOR-DETERMINISM",
        "CAL-COLLISION",
        "CAL-INVERSE-METADATA",
        "CAL-CHECKPOINT",
        "CAL-EXPOSURE",
        "CAL-NULL-STATES",
    }
