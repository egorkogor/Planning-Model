from __future__ import annotations

import inspect
import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from analysis.sample_size import calculate_components_from_structured_requirements
from validation import phase_check_runner
from validation.implementation_audit_validator import REQUIRED as IMPLEMENTATION_CHECKS
from validation.role_validator import validate_role_independence

ROOT = Path(__file__).resolve().parents[1]


def y(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_stage1b_reserve_covers_locked_power_scale() -> None:
    split = y("docs/data/dataset_split_contract_v1.yaml")
    row = split["partitions"]["stage1b_confirmatory_reserve"]
    assert row["target_base_tasks"] == 4000
    assert sum(row["quota_by_block_count"].values()) == 4000


def test_planner_sample_size_omits_nonexistent_tost_component() -> None:
    stats = y("docs/statistics/statistics_contract_v1.yaml")
    assert stats["sample_size"]["component_set_by_stage"]["PLANNER"] == [
        "primary_ci", "primary_power", "current_vs_shuffled_power"
    ]
    assert "equivalence_TOST_power" not in stats["sample_size"]["component_set_by_stage"]["PLANNER"]


def test_structured_calculator_accepts_stage_specific_subset() -> None:
    rows = [{"pair_id": f"p{i}", "left": i % 2, "right": (i // 2) % 2, "difference": (i % 2) - ((i // 2) % 2)} for i in range(20)]
    req = {name: {"comparison": name, "analysis_type": "TASK_PAIRED", "complete_pair_count": 20, "pairs": rows} for name in ("primary_ci", "primary_power", "current_vs_shuffled_power")}
    out = calculate_components_from_structured_requirements(req, simulations=2, ci_resamples=10, maximum_n=300, minimum_n=300, target_power=0.0, power_confirmation_points=1)
    assert "equivalence_TOST_power" not in out


def test_g06_implementation_audit_is_trusted_core_check() -> None:
    contract = y("docs/operator/phase_check_contract_v1.yaml")
    assert contract["core_checks"]["P06_pre_06"] == "implementation_audit"
    assert contract["core_checks"]["P06_pre_07"] == "implementation_audit"
    phase = next(p for p in y("docs/operator/phase_registry_v1.yaml")["phases"] if p["phase_id"] == "P06")
    assert "reports/independent-implementation-audit.json" in phase["pre_gate_required_outputs"]
    assert len(IMPLEMENTATION_CHECKS) == 15


def test_phase_runner_no_longer_has_divergent_script_copy() -> None:
    wrapper = (ROOT / "scripts/phase_check_runner.py").read_text(encoding="utf-8")
    assert "from validation.phase_check_runner import main" in wrapper
    assert "def core_check" not in wrapper


def test_runtime_stack_lock_is_schema_bound_and_core_checked() -> None:
    contract = y("docs/operator/phase_check_contract_v1.yaml")
    assert contract["core_checks"]["P02_exec_05"] == "runtime_stack_lock"
    source = inspect.getsource(phase_check_runner.core_check)
    assert "runtime_stack_lock.schema.json" in source
    assert "offline_cache_manifest_sha256" in json.loads((ROOT / "docs/schemas/runtime_stack_lock.schema.json").read_text())["properties"]["model_runtime"]["required"]


def test_all_execution_environments_must_be_unique() -> None:
    def role(name: str, key: str, kind: str, family: str = "f") -> dict:
        return {"role": name, "reviewer_type": kind, "agent_provider": "p", "model_family": family, "model_revision": "r", "system_prompt_hash": "sha256:" + "1" * 64, "environment_identity": key + "-env", "credential_principal": key, "public_signing_key_id": key + "-key"}
    plan = {
        "builder": role("BUILDER", "b", "MODEL"),
        "data_sealer": role("DATA_SEALER", "s", "SERVICE_PROCESS"),
        "evaluator": role("EVALUATION_RUNNER", "e", "SERVICE_PROCESS"),
        "auditor": role("AUDITOR", "a", "MODEL", "other"),
        "statistical_reviewer": role("STATISTICAL_REVIEWER", "r", "MODEL", "other2"),
    }
    assert not validate_role_independence(plan)
    plan["evaluator"]["environment_identity"] = plan["data_sealer"]["environment_identity"]
    assert any("unique" in err for err in validate_role_independence(plan))


def test_implementation_candidate_binds_both_audits_and_commit() -> None:
    schema = json.loads((ROOT / "docs/schemas/implementation_lock_candidate.schema.json").read_text())
    assert "statistical_audit_sha256" in schema["required"]
    assert "implementation_audit_sha256" in schema["required"]
    assert "compute_profile_sha256" in schema["required"]
    contract = y("docs/operator/phase_check_contract_v1.yaml")
    assert contract["core_checks"]["P06_pre_02"] == "implementation_candidate"
    source = (ROOT / "validation/implementation_candidate_validator.py").read_text()
    assert "audit.get('reviewed_commit')!=commit" in source
    assert "implementation_audit_sha256" in source
    assert "statistical_audit_sha256" in source
    assert "compute profile hash mismatch" in source
