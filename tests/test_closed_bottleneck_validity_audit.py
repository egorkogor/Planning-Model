from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / ".research/audits/closed_bottleneck_validity_audit_v1.yaml"


def _audit() -> dict:
    return yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8"))


def test_claims_are_decomposed_and_not_collapsed() -> None:
    audit = _audit()
    claims = audit["claim_decomposition"]
    assert claims["mechanical_channel_closure"]["required_for_future_claim"] is True
    assert claims["allowed_channel_usability"]["required_for_future_claim"] is True
    assert claims["semantic_content_necessity"]["required_for_future_claim"] is True
    assert claims["collapse_into_single_claim"] == "FORBIDDEN"
    assert "not semantic necessity" in claims["mechanical_channel_closure"]["interpretation"]
    usability = claims["allowed_channel_usability"]["interpretation"]
    assert "not normal trained-model reliance" in usability


def test_stateless_boundary_resets_all_prohibited_cross_step_state() -> None:
    audit = _audit()
    boundary = audit["stateless_boundary"]
    reset = boundary["per_step_reset"]
    assert boundary["required"] is True
    assert reset["model_hidden_state"] == "FRESH_OR_EMPTY"
    assert reset["kv_cache"] == "FRESH_OR_EMPTY"
    assert reset["autoregressive_token_history"] == "ABSENT"
    failures = " ".join(boundary["fail_closed_conditions"])
    for token in ("hidden state", "KV cache", "token IDs", "undeclared cross-step object"):
        assert token in failures


def test_allowed_context_manifest_is_exact_and_excludes_history_bypass() -> None:
    audit = _audit()
    manifest = audit["allowed_context_manifest"]
    fields = set(manifest["exact_fields"])
    assert {
        "task_encoding_hash",
        "initial_public_state_hash",
        "planning_state_mode",
        "entity_catalog_hash",
        "tool_catalog_hash",
        "experiment_config_hash",
        "model_checkpoint_identity",
        "position_index",
    } <= fields
    forbidden = " ".join(manifest["forbidden_fields"])
    assert "prior natural-language reasoning text" in forbidden
    assert "prior generated token IDs" in forbidden
    assert "prior hidden states" in forbidden
    assert "prior KV cache" in forbidden
    assert "future/gold action" in forbidden


def test_refs_and_tools_cannot_encode_operator_or_argument_roles() -> None:
    audit = _audit()
    interface = audit["concept_interface"]["fields"]
    refs = " ".join(interface["grounding_refs"]["requirements"])
    tool = " ".join(interface["tool_ref"]["requirements"])
    assert "no operator/action type" in refs
    assert "no argument roles" in refs
    assert "no relations" in refs
    assert "no arguments" in tool
    assert "no operator/argument-role encoding" in tool
    assert audit["concept_interface"]["per_episode_permutation_required"] is True


def test_metadata_only_baseline_uses_exact_nonconcept_inputs() -> None:
    audit = _audit()
    baseline = audit["metadata_only_baseline"]
    assert baseline["required"] is True
    assert baseline["concept_fields_removed"] == ["z_semantic"]
    assert baseline["no_extra_history"] is True
    assert "exactly the same allowed non-concept manifest" in baseline["inputs"]
    assert "cannot support a closed semantic-bottleneck claim" in baseline["invalidity_rule"]


def test_history_collision_makes_replanning_non_discriminative() -> None:
    audit = _audit()
    collision = audit["history_collision_counterfactual"]
    construction = collision["construction"]
    assert collision["required"] is True
    assert construction["allowed_nonconcept_context"] == "BYTE_IDENTICAL_ACROSS_PAIR"
    assert construction["grounding_refs_and_tool_ref"] == "BYTE_IDENTICAL_ACROSS_PAIR"
    assert construction["concept_content"] == "DIFFERENT"
    assert construction["required_next_action"] == "DIFFERENT"
    assert "re-planning" in collision["purpose"]


def test_oracle_proves_usability_not_trained_model_reliance() -> None:
    audit = _audit()
    oracle = audit["counterfactual_oracle_control"]
    assert oracle["required"] is True
    assert oracle["construction"]["allowed_nonconcept_context"] == "BYTE_IDENTICAL"
    success = oracle["interpretation"]["success"]
    assert success == "The resolver can use the allowed concept channel."
    assert oracle["interpretation"]["forbidden_inference"] == (
        "The trained model normally uses the channel."
    )


def test_wrong_concept_donor_is_required_beyond_zero_shuffle() -> None:
    audit = _audit()
    interventions = audit["content_interventions"]
    assert "out of distribution" in interventions["zero"]["warning"]
    assert "distributionally implausible" in interventions["shuffle"]["warning"]
    donor = interventions["matched_in_distribution_wrong_concept_donor"]
    assert donor["required"] is True
    assert "same checkpoint" in donor["matching_requirements"]
    assert "different semantic content" in donor["matching_requirements"]


def test_forbidden_channel_canary_fails_closed() -> None:
    audit = _audit()
    canary = audit["forbidden_channel_canary"]
    assert canary["required"] is True
    assert canary["target_generation"]["balanced"] is True
    assert canary["target_generation"]["unpredictable_from_allowed_metadata"] is True
    assert canary["failure"] == "INVALID_MECHANICAL_CLOSURE"
    assert "reused KV cache" in canary["examples_of_prohibited_carrier"]


def test_static_fixture_matrix_covers_all_required_adversarial_roles() -> None:
    audit = _audit()
    kinds = {item["kind"] for item in audit["static_fixture_matrix"]}
    assert {
        "mechanical_closure",
        "metadata_leakage",
        "replanning_shortcut",
        "necessity_fixture",
        "allowed_channel_positive_control",
        "content_intervention",
    } <= kinds
    ids = {item["id"] for item in audit["static_fixture_matrix"]}
    assert {
        "STATeless-boundary-negative-kv",
        "metadata-only-no-concept",
        "history-collision-pair",
        "oracle-concept-counterfactual",
        "wrong-concept-donor",
        "forbidden-channel-canary",
    } <= ids


def test_audit_remains_non_outcome_bearing_and_does_not_authorize_a3b() -> None:
    audit = _audit()
    assert audit["status"] == "ADVISORY_DISCOVERY_DESIGN_NON_OUTCOME_BEARING"
    assert audit["authority"]["frozen_v1_21_read_only"] is True
    boundary = audit["promotion_boundary"]
    assert boundary["scientific_execution"] == "FORBIDDEN"
    assert boundary["concept_bottleneck_implementation"] == "NOT_AUTHORIZED"
    assert boundary["real_model_discovery"] == "FUTURE_SEPARATE_REVIEWER_AUTHORIZATION_REQUIRED"
    assert boundary["go_latent"] == "NOT EVALUATED"
