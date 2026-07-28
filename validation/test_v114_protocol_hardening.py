from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _yaml(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_scientific_lock_covers_full_plan_and_capacity_semantics() -> None:
    paths = set(_yaml("docs/operator/scientific_lock_v1.yaml")["protected_paths"])
    required = {
        "docs/schemas/attempt_log.schema.json",
        "docs/schemas/episode_log.schema.json",
        "docs/schemas/episode_plan_manifest.schema.json",
        "docs/schemas/full_plan_lineage_index.schema.json",
        "docs/schemas/compute_profile.schema.json",
        "docs/schemas/capacity_preflight.schema.json",
        "validation/full_plan_lineage_validator.py",
        "validation/capacity_validator.py",
    }
    assert required <= paths


def test_every_confirmatory_freeze_binds_compute_and_capacity() -> None:
    schema = json.loads((ROOT / "docs/schemas/experiment_freeze.schema.json").read_text(encoding="utf-8"))
    assert {"compute_profile_sha256", "capacity_preflight_sha256"} <= set(schema["required"])
    source = (ROOT / "validation/confirmatory_lineage_validator.py").read_text(encoding="utf-8")
    assert '"compute_profile_sha256": "reports/compute-profile.json"' in source
    assert '"capacity_preflight_sha256": "reports/preflight-final.json"' in source


def test_implementation_candidate_binds_compute_profile_explicitly() -> None:
    schema = json.loads((ROOT / "docs/schemas/implementation_lock_candidate.schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == "work-planner-implementation-candidate/1.3"
    assert "compute_profile_sha256" in schema["required"]


def test_confirmatory_tost_is_not_an_active_rule_or_component() -> None:
    contract = _yaml("docs/statistics/statistics_contract_v1.yaml")
    assert contract["historical_compatibility"]["confirmatory_use"] == "forbidden"
    assert "equivalence_margin" not in contract["sample_size"]
    active_text = (ROOT / "validation/statistics_validator.py").read_text(encoding="utf-8")
    assert "forbidden confirmatory paired_tost rule" in active_text
    for stage, components in contract["sample_size"]["component_set_by_stage"].items():
        assert all("tost" not in name.lower() and "equivalence" not in name.lower() for name in components), stage


def test_stage1b_selection_is_task_only_and_never_uses_arm_outputs() -> None:
    sealing = _yaml("docs/controls/confirmatory_sealing_contract_v1.yaml")
    certification = sealing["sealer_protocol"]["stage1b_hidden_control_certification"]
    forbidden = " ".join(certification["forbidden_selection_inputs"]).lower()
    assert "planner" in forbidden and "llm" in forbidden and "arm outcomes" in forbidden
    checks = " ".join(certification["checks"]).lower()
    assert "retain every plan-generation" in checks
    assert "never exclude" in checks


def test_flops_budget_exhaustion_has_a_typed_failure_code() -> None:
    common = json.loads((ROOT / "docs/schemas/common.schema.json").read_text(encoding="utf-8"))
    assert "FLOPS_BUDGET_EXHAUSTED" in common["$defs"]["issue_code"]["enum"]


def test_stage1b_selection_contract_has_no_reachable_state_or_planner_support_veto() -> None:
    intent = _yaml("docs/controls/intent_control_contract_v1.yaml")
    assert intent["scope"]["stage"] == "STAGE1A_INTERFACE"
    assert "forbidden" in intent["scope"]["stage1b_use"]
    phase = _yaml("docs/operator/phase_registry_v1.yaml")
    p15 = next(row for row in phase["phases"] if row["phase_id"] == "P15")
    text = " ".join(p15["actions"] + [c["description"] for c in p15["execution_checks"]]).lower()
    assert "reachable-state" not in text and "total control coverage" not in text
    assert "task-only" in text and "planner" in text
    sealer_schema = json.loads((ROOT / "docs/schemas/sealer_manifest.schema.json").read_text(encoding="utf-8"))
    cert = sealer_schema["properties"]["control_certification"]
    assert cert["properties"]["selection_basis"]["const"] == "TASK_AND_DOMAIN_METADATA_ONLY"
    assert cert["properties"]["plan_or_control_degeneracy_exclusion_count"]["const"] == 0


def test_random_codebook_schema_threshold_matches_locked_contract() -> None:
    contract = _yaml("docs/controls/random_codebook_contract_v1.yaml")
    schema = json.loads((ROOT / "docs/schemas/random_codebook_manifest.schema.json").read_text(encoding="utf-8"))
    threshold = contract["quality_checks"]["maximum_abs_pairwise_cosine"]
    assert threshold == 0.3
    assert schema["properties"]["maximum_abs_pairwise_cosine"]["maximum"] == threshold
    assert contract["quality_checks"]["prelock_exhaustive_schema_universe_max_abs_pairwise_cosine"] <= threshold


def test_implementation_audit_contract_schema_and_validator_have_one_exact_check_set() -> None:
    import validation.implementation_audit_validator as validator
    contract = _yaml("docs/audit/independent_audit_contract_v1.yaml")
    schema = json.loads((ROOT / "docs/schemas/implementation_audit.schema.json").read_text(encoding="utf-8"))
    contract_ids = set(contract["implementation_audit"]["required_checks"])
    schema_ids = set(schema["properties"]["checks"]["items"]["properties"]["check_id"]["enum"])
    assert contract_ids == schema_ids == validator.REQUIRED
    assert len(contract_ids) == 16
    assert "TASK_ONLY_SELECTION_CERTIFICATION_ENGINE" in contract_ids
    assert "CONTROL_CERTIFICATION_ENGINE" not in contract_ids
