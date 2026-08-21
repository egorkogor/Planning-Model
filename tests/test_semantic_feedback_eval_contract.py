from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / ".research/evals/semantic_feedback_causal_eval_v1.yaml"


def _contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def _synthetic_status(case: dict) -> str:
    if not case["checkpoint_match"]:
        return "INVALID_CHECKPOINT_OR_RETRAINING"
    if not case["shortcut_controls_pass"]:
        return "INVALID_SHORTCUT_NOT_EXCLUDED"
    if case["reference_exposure_count"] == 0:
        return "NOT_EVALUATED_NO_DOWNSTREAM_OPPORTUNITY"
    if not case["intervention_applicable"]:
        return "NOT_EVALUATED_INTERVENTION_NOT_APPLICABLE"
    if case.get("treatment_terminated_before_planned_free_running_position", False):
        return "CENSORED_TERMINATED_BEFORE_PLANNED_INTERVENTION"
    if not case["endpoint_defined"]:
        return "NOT_APPLICABLE_ENDPOINT_UNDEFINED"
    return "EVALUATED"


def test_three_estimands_are_distinct_and_not_collapsed() -> None:
    contract = _contract()
    estimands = {item["id"]: item for item in contract["estimands"]}
    assert set(estimands) == {
        "EST-PLN-SEM-ARCH-BUNDLE",
        "EST-PLN-SEM-GEOMETRY",
        "EST-PLN-SEM-RELIANCE",
    }
    assert estimands["EST-PLN-SEM-ARCH-BUNDLE"]["contrasts"] == ["A3_vs_A2c"]
    assert estimands["EST-PLN-SEM-GEOMETRY"]["contrasts"] == ["A3_vs_A3r"]
    assert estimands["EST-PLN-SEM-RELIANCE"]["contrasts"] == ["A3_vs_A4", "A3_vs_A5"]
    assert contract["construct"]["collapse_into_single_causal_contrast"] == "FORBIDDEN"
    assert contract["triangulation"]["pooled_semantic_effect"] == "FORBIDDEN"


def test_estimand_interpretation_boundaries_match_legacy_semantics() -> None:
    contract = _contract()
    estimands = {item["id"]: item for item in contract["estimands"]}
    bundle = " ".join(estimands["EST-PLN-SEM-ARCH-BUNDLE"]["forbidden_interpretation"])
    geometry = " ".join(estimands["EST-PLN-SEM-GEOMETRY"]["forbidden_interpretation"])
    reliance = " ".join(estimands["EST-PLN-SEM-RELIANCE"]["forbidden_interpretation"])
    assert "pure semantic representation effect" in bundle
    assert "random code is information-free" in geometry
    assert "A3 is superior to A2c" in reliance


def test_mandatory_validity_controls_are_present() -> None:
    contract = _contract()
    controls = contract["interventions_and_controls"]
    assert controls["zero_intervention"]["required"] is True
    assert controls["constrained_shuffle"]["required"] is True
    assert controls["matched_in_distribution_wrong_semantic_donor"]["required"] is True
    assert controls["random_codebook_robustness"]["required_for_estimand"] == (
        "EST-PLN-SEM-GEOMETRY"
    )
    assert controls["compositional_novel_signature_stress"]["required_for_estimand"] == (
        "EST-PLN-SEM-GEOMETRY"
    )
    assert controls["zero_intervention"]["checkpoint"] == "exact_A3"
    assert controls["constrained_shuffle"]["checkpoint"] == "exact_A3"
    assert controls["matched_in_distribution_wrong_semantic_donor"]["checkpoint"] == "exact_A3"


def test_shortcut_defenses_and_fixed_prefix_probe_are_mandatory() -> None:
    contract = _contract()
    shortcuts = contract["shortcut_defenses"]
    assert shortcuts["task_id_ref_signature_baseline"]["required"] is True
    assert shortcuts["opaque_permuted_ids"]["required"] is True
    assert shortcuts["metadata_counterfactual_decoupling"]["required"] is True
    probe = contract["fixed_prefix_first_feedback_probe"]
    assert probe["required"] is True
    assert probe["immediate_endpoint"]["name"] == "immediate_next_step_effect"
    assert probe["free_running_endpoint"]["name"] == "free_running_total_effect"
    assert probe["immediate_and_total_must_not_be_collapsed"] is True


def test_exposure_and_censoring_fail_closed() -> None:
    contract = _contract()
    exposure = contract["exposure_and_censoring"]
    assert exposure["downstream_exposure_precondition"] is True
    assert exposure["post_treatment_survivor_denominator"] == "FORBIDDEN"
    assert exposure["zero_reference_exposure"] == "INVALID_FOR_ESTIMAND"
    assert exposure["null_is_not_zero_effect"] is True
    for estimand in contract["estimands"]:
        assert estimand["downstream_feedback_exposure_required"] is True
        assert estimand["zero_exposure_semantics"] == "INVALID_FOR_ESTIMAND"


def test_same_checkpoint_and_no_retraining_are_bound() -> None:
    contract = _contract()
    binding = contract["checkpoint_and_provenance"]
    assert binding["A3_A4_A5_exact_checkpoint_identity"] == "MANDATORY"
    assert binding["A4_A5_retraining"] == "FORBIDDEN"
    assert binding["same_checkpoint_content_interventions_retraining"] == "FORBIDDEN"


def test_synthetic_calibration_classifies_null_censoring_and_invalidity() -> None:
    contract = _contract()
    cases = contract["synthetic_calibration"]["cases"]
    for case in cases:
        assert _synthetic_status(case) == case["expected_status"], case["id"]
        if case["expected_status"] != "EVALUATED":
            assert case.get("expected_endpoint") is None


def test_calibration_covers_adversarial_construct_cases() -> None:
    contract = _contract()
    kinds = {case["kind"] for case in contract["synthetic_calibration"]["cases"]}
    assert {
        "positive_control",
        "matched_wrong_semantic",
        "shortcut_invariance",
        "shortcut_counterfactual",
        "null_semantics",
        "invalidity",
        "censoring",
        "random_code_lookup_stress",
    } <= kinds


def test_contract_remains_non_outcome_bearing_and_legacy_read_only() -> None:
    contract = _contract()
    authority = contract["authority"]
    assert authority["evidence_class"] == "DISCOVERY_DESIGN_NON_OUTCOME_BEARING"
    assert contract["legacy_bindings"]["frozen_v1_21_read_only"] is True
    assert contract["synthetic_calibration"]["scientific_runtime_invoked"] is False
    assert contract["synthetic_calibration"]["outcome_bearing"] is False
    assert contract["promotion_boundary"]["confirmatory_freeze"] == "NOT_AUTHORIZED"
    assert contract["promotion_boundary"]["go_latent"] == "NOT EVALUATED"
