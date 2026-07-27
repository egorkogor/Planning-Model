from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_outcome_relevant_architecture_domain_and_statistics_are_scientifically_locked():
    paths = set(load_yaml("docs/operator/scientific_lock_v1.yaml")["protected_paths"])
    required = {
        "docs/domain/**",
        "docs/architecture/planner_architecture_v1.yaml",
        "docs/architecture/task_encoding_v1.yaml",
        "docs/architecture/a1_token_grammar_v1.yaml",
        "docs/semantic/semantic_target_v1.yaml",
        "analysis/decision_gates.py",
        "analysis/sample_size.py",
        "validation/statistics_validator.py",
        "requirements.lock",
        ".github/workflows/ci.yml",
    }
    assert required <= paths


def test_implementation_only_patch_allowlist_is_narrow_and_excludes_scientific_code():
    contract = load_yaml("docs/operator/implementation_lock_v1.yaml")
    allowed = contract["pre_lock_patch_window"]["allowed_path_globs"]
    assert "analysis/**" not in allowed
    assert "validation/**" not in allowed
    assert "docs/architecture/**" not in allowed
    assert "docs/domain/**" not in allowed
    assert "src/planner_llm_mvp/runtime/**" in allowed


def test_machine_gates_include_seed_direction_positions_and_stage_validity():
    gates = load_yaml("docs/statistics/statistics_contract_v1.yaml")["stage_gates"]
    planner = {g["gate_id"] for g in gates["PLANNER"]["architecture_gates"]}
    assert "GO_TYPED_POSITION_REDUCTION" in planner
    assert {
        "SEED_DIRECTION_A2B_A2",
        "SEED_DIRECTION_A2C_A2B",
        "SEED_DIRECTION_A3_A2C",
        "SEED_DIRECTION_A3_A4",
        "SEED_DIRECTION_A3_A5",
    } <= planner
    required_validity = {
        "PARSE_FAILURE_FLOOR",
        "INFRASTRUCTURE_FAILURE_FLOOR",
        "INCOMPLETE_PAIR_FLOOR",
        "CONTRACT_VIOLATION_ZERO",
        "HASH_VIOLATION_ZERO",
        "DETERMINISM_VIOLATION_ZERO",
    }
    assert required_validity <= {g["gate_id"] for g in gates["STAGE1A"]["core_gates"]}
    assert required_validity <= {g["gate_id"] for g in gates["STAGE1B"]["core_gates"]}


def test_sample_size_contract_uses_final_estimator_structures():
    contract = load_yaml("docs/statistics/statistics_contract_v1.yaml")["sample_size"]
    assert contract["method"] == "estimator_matched_paired_binary_simulation_v5"
    assert contract["estimator_match_required"] is True
    assert contract["power_confidence_level"] == 0.95
    assert contract["power_confirmation_points"] == 2
    schema = json.loads((ROOT / "docs/schemas/sample_size_input.schema.json").read_text())
    defs = schema["$defs"]
    assert "seed_groups" in defs["planner"]["properties"]
    assert "task_clusters" in defs["stage1a"]["properties"]
    assert "pairs" in defs["stage1b"]["properties"]


def test_clean_runtime_dependency_lock_contains_statistics_and_crypto_packages():
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    assert "scipy==" in lock
    assert "cryptography==" in lock
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'python-version: ["3.11", "3.13"]' in ci
    assert "git diff --exit-code" in ci


def test_role_registry_requires_challenge_response_attestation():
    schema = json.loads((ROOT / "docs/schemas/public_key_registry.schema.json").read_text())
    required = set(schema["properties"]["keys"]["items"]["required"])
    assert {
        "challenge_nonce_b64",
        "challenge_signature_b64",
        "environment_identity_sha256",
        "credential_principal_sha256",
        "attested_at",
    } <= required


def test_p_replay_uses_two_precomputed_full_plan_contexts_without_replanning():
    contract = load_yaml("docs/controls/p_replay_contract_v1.yaml")
    assert set(contract["contexts"]) == {"PLANNER_CONFIRMATORY_A3", "STAGE1B_E1"}
    assert contract["shared_invariants"]["plan_generation_before_execution"] == "required"
    assert contract["shared_invariants"]["planner_calls_during_execution"] == 0
    assert contract["shared_invariants"]["plan_patch_or_regeneration"] == "forbidden"
    assert contract["contexts"]["PLANNER_CONFIRMATORY_A3"]["phase"] == "P08"
    assert contract["contexts"]["STAGE1B_E1"]["phase"] == "P17"
