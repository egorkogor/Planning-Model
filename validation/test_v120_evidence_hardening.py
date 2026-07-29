from __future__ import annotations

import hashlib
import json
from pathlib import Path
from copy import deepcopy

import numpy as np

from docs.domain.intent_labeler_v1 import label_intent
from validation.hashing import hash_json
from validation.planner_evidence_validator import (
    TRAINABLE_VARIANTS,
    canonical_initialized_array,
    validate_checkpoint_manifest,
    validate_model_audit_check_evidence,
)
from validation.test_v118_architecture_evidence import (
    _prepare_contracts,
    _sha,
    _write_final_matrix,
    _write_safetensors,
)


def _evidence(check_id: str, bindings: list[dict], details: dict, value: dict) -> dict:
    obj = {
        "schema_version": "work-planner-model-audit-evidence/1.0",
        "run_id": "run-1",
        "check_id": check_id,
        "status": "PASS",
        "recomputed_value": value,
        "expected_value": value,
        "bindings": bindings,
        "details": details,
    }
    obj["evidence_hash"] = hash_json(obj)
    return obj


def _binding(root: Path, rel: str) -> dict:
    return {"path": rel, "sha256": _sha(root / rel)}


def test_parameter_tolerance_is_recomputed_from_inventory(tmp_path: Path) -> None:
    inventory, _ = _prepare_contracts(tmp_path)
    value = {
        "inventory_sha256": inventory["inventory_hash"],
        "common_superset_parameter_count": 1,
        "active_parameter_counts": {variant: 1 for variant in TRAINABLE_VARIANTS},
        "total_parameter_count_tolerance_fraction": 0.0,
        "all_trainable_arms_share_exact_inventory": True,
    }
    obj = _evidence(
        "PARAMETER_TOLERANCE",
        [
            _binding(tmp_path, "docs/architecture/planner_module_inventory_v1.yaml"),
            _binding(tmp_path, "reports/model-evidence/parameter-inventory.json"),
        ],
        {},
        value,
    )
    assert validate_model_audit_check_evidence(tmp_path, obj) == []
    bad = deepcopy(obj)
    bad["recomputed_value"]["common_superset_parameter_count"] = 2
    bad["evidence_hash"] = hash_json({k: v for k, v in bad.items() if k != "evidence_hash"})
    assert any("validator recomputation" in e for e in validate_model_audit_check_evidence(tmp_path, bad))


def test_same_information_is_recomputed_with_normative_labeler(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "docs/domain/intent_labeler_v1.py", "docs/domain/intent_catalog_v1.yaml",
        "docs/semantic/semantic_target_v1.yaml",
    ):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((root / rel).read_bytes())
    templates = [
        {"state": [["ON", "@B2", "@B0"], ["CLEAR", "@B2"], ["HAND_EMPTY"]], "goal": [["ON", "@B0", "@B1"]], "action": {"action": "UNSTACK", "args": ["@B2", "@B0"]}, "distance": 4},
        {"state": [["ON", "@B2", "@B1"], ["CLEAR", "@B2"], ["HAND_EMPTY"]], "goal": [["ON", "@B0", "@B1"]], "action": {"action": "UNSTACK", "args": ["@B2", "@B1"]}, "distance": 4},
        {"state": [["ON_TABLE", "@B0"], ["CLEAR", "@B0"], ["HAND_EMPTY"]], "goal": [["ON", "@B0", "@B1"]], "action": {"action": "PICK_UP", "args": ["@B0"]}, "distance": 2},
        {"state": [["HOLDING", "@B2"], ["ON_TABLE", "@B0"], ["CLEAR", "@B0"]], "goal": [["ON", "@B0", "@B2"]], "action": {"action": "PUT_DOWN", "args": ["@B2"]}, "distance": 3},
        {"state": [["HOLDING", "@B0"], ["ON_TABLE", "@B1"], ["CLEAR", "@B1"]], "goal": [["ON", "@B0", "@B1"]], "action": {"action": "STACK", "args": ["@B0", "@B1"]}, "distance": 1},
        {"state": [["HOLDING", "@B0"], ["ON_TABLE", "@B1"], ["CLEAR", "@B1"], ["ON_TABLE", "@B2"], ["CLEAR", "@B2"]], "goal": [["ON", "@B0", "@B2"]], "action": {"action": "STACK", "args": ["@B0", "@B1"]}, "distance": 3},
        {"state": [["ON_TABLE", "@B0"], ["CLEAR", "@B0"], ["HAND_EMPTY"]], "goal": [["ON_TABLE", "@B0"]], "action": {"action": "END", "args": []}, "distance": 0},
    ]
    inputs = []
    for intent_id, template in enumerate(templates):
        for repetition in range(3):
            action = template["action"]
            case = {
                "case_id": f"intent-{intent_id}-{repetition}",
                "state": template["state"], "goal": template["goal"],
                "all_shortest_first_actions": [action], "selected_action": action,
                "remaining_oracle_length": template["distance"],
            }
            out = label_intent(case["state"], case["goal"], case["all_shortest_first_actions"], case["selected_action"], case["remaining_oracle_length"])
            assert out["intent_id"] == intent_id
            case["a2c_semantic_signature"] = out["semantic_signature"]
            case["a3_canonical_text"] = out["canonical_text"]
            inputs.append(case)
    counts = {str(intent_id): 3 for intent_id in range(7)}
    value = {"case_count": 21, "mismatch_count": 0, "intent_counts": counts}
    obj = _evidence(
        "SAME_INFORMATION",
        [_binding(tmp_path, "docs/domain/intent_labeler_v1.py"), _binding(tmp_path, "docs/domain/intent_catalog_v1.yaml"), _binding(tmp_path, "docs/semantic/semantic_target_v1.yaml")],
        {"cases": inputs}, value,
    )
    assert validate_model_audit_check_evidence(tmp_path, obj) == []
    bad = deepcopy(obj)
    bad["details"]["cases"][0]["a3_canonical_text"] = "fabricated"
    bad["evidence_hash"] = hash_json({k: v for k, v in bad.items() if k != "evidence_hash"})
    assert any("A3 canonical text mismatch" in e for e in validate_model_audit_check_evidence(tmp_path, bad))


def test_raw_rollout_invariants_are_recomputed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bindings = []
    for rel in ("docs/architecture/a1_token_grammar_v1.yaml", "docs/training/planner_training_contract_v1.yaml"):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((root / rel).read_bytes())
        bindings.append(_binding(tmp_path, rel))
    cases = []
    for variant in ("A1", "A2", "A2b", "A2c", "A3", "A3r"):
        for intent_id in range(6):
            case_id = f"{variant}-{intent_id}"
            case = {
                "case_id": case_id, "variant": variant, "intent_id": intent_id,
                "planner_calls": 1, "execution_planner_calls": 0,
                "domain_action_mask_applied": False, "grammar_constrained_decoding": False,
                "replanning": False, "plan_positions_consumed": intent_id + 1,
            }
            for field, suffix, payload in (
                ("raw_logits", "logits.bin", b"logits"),
                ("frozen_plan", "plan.json", b"{}"),
                ("event_log", "events.jsonl", b"{}\n"),
            ):
                rel = f"reports/model-evidence/raw-rollout/{case_id}.{suffix}"
                path = tmp_path / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload + case_id.encode())
                case[f"{field}_path"] = rel
                case[f"{field}_sha256"] = _sha(path)
            cases.append(case)
    value = {"case_count": 36, "violation_count": 0, "matrix_complete": True}
    obj = _evidence("RAW_ROLLOUT", bindings, {"cases": cases}, value)
    assert validate_model_audit_check_evidence(tmp_path, obj) == []
    bad = deepcopy(obj)
    bad["details"]["cases"][1]["replanning"] = True
    bad["evidence_hash"] = hash_json({k: v for k, v in bad.items() if k != "evidence_hash"})
    assert any("replanning mismatch" in e for e in validate_model_audit_check_evidence(tmp_path, bad))

def test_dormant_gradient_check_loads_all_six_audits(tmp_path: Path) -> None:
    inventory, _ = _prepare_contracts(tmp_path)
    bindings = [_binding(tmp_path, "reports/model-evidence/parameter-inventory.json")]
    for variant in TRAINABLE_VARIANTS:
        audit = {
            "schema_version": "work-planner-dormant-gradient-audit/1.0", "run_id": "run-1",
            "variant": variant, "seed": 17, "parameter_inventory_sha256": inventory["inventory_hash"],
            "batches_audited": 2,
            "entries": [{
                "name": "weight", "expected_activity": "ACTIVE", "gradient_state": "PRESENT_FINITE",
                "nonfinite_count": 0, "materialized_zero_count": 0,
            }],
            "status": "PASS",
        }
        audit["audit_hash"] = hash_json(audit)
        rel = f"reports/model-evidence/dormant-gradients/{variant}-seed-17.json"
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit))
        bindings.append(_binding(tmp_path, rel))
    value = {"audit_seed": 17, "variants_passed": list(TRAINABLE_VARIANTS), "required_variant_count": 6}
    obj = _evidence("DORMANT_GRADIENT", bindings, {}, value)
    assert validate_model_audit_check_evidence(tmp_path, obj) == []
    (tmp_path / "reports/model-evidence/dormant-gradients/A3-seed-17.json").unlink()
    assert validate_model_audit_check_evidence(tmp_path, obj)


def test_checkpoint_values_reject_zero_or_unchanged_artifacts(tmp_path: Path) -> None:
    _write_final_matrix(tmp_path)
    cp_path = tmp_path / "reports/training/checkpoints/final-A3-seed-101.json"
    cp = json.loads(cp_path.read_text())
    model_path = tmp_path / cp["model_file_path"]
    inv = json.loads((tmp_path / cp["parameter_inventory_manifest_path"]).read_text())
    initial = canonical_initialized_array(inv["tensors"][0], 101)
    metadata = {
        "run_id": "run-1", "checkpoint_kind": "FINAL_EQUAL_DATA", "variant": "A3",
        "seed": "101", "optimizer_step": "12000", "parameter_inventory_sha256": inv["inventory_hash"],
    }
    _write_safetensors(model_path, [{"name": "weight", "shape": [1]}], metadata, {"weight": initial})
    cp["model_file_sha256"] = _sha(model_path)
    cp["manifest_hash"] = hash_json({k: v for k, v in cp.items() if k != "manifest_hash"})
    assert any("no changed active tensor" in e for e in validate_checkpoint_manifest(tmp_path, cp))

    opt_path = tmp_path / cp["optimizer_state_path"]
    _write_safetensors(
        opt_path,
        [{"name": "weight.exp_avg", "shape": [1]}, {"name": "weight.exp_avg_sq", "shape": [1]}],
        {"optimizer_step": "12000", "variant": "A3", "optimizer": "AdamW", "beta1": "0.9", "beta2": "0.95", "eps": "1e-08", "weight_decay": "0.01"},
    )
    cp["optimizer_state_sha256"] = _sha(opt_path)
    cp["manifest_hash"] = hash_json({k: v for k, v in cp.items() if k != "manifest_hash"})
    assert any("all zero" in e for e in validate_checkpoint_manifest(tmp_path, cp))


def test_p08_rejects_noncanonical_checkpoint_link(tmp_path: Path) -> None:
    from validation.full_plan_lineage_validator import validate_lineage_index
    from validation.test_v114_full_plan_lineage import _planner_confirmatory_matrix_rows, _selection_fields

    rows = _planner_confirmatory_matrix_rows(tmp_path)
    index = {
        "schema_version": "work-planner-lineage/1.0", "run_id": "run-1", "stage": "PLANNER",
        "compute_profile": None, "compute_profile_sha256": None,
        "records": rows, "index_hash": "sha256:" + "1" * 64,
        **_selection_fields(tmp_path, "PLANNER", ["bw-00000001"]),
    }
    target = next(r for r in index["records"] if r["arm"] == "PLANNER_A3_RAW" and r["planner_seed"] == 101)
    canonical = tmp_path / target["checkpoint_manifest"]
    link = tmp_path / "reports/training/checkpoints/link-fabricated.json"
    link.write_bytes(canonical.read_bytes())
    target["checkpoint_manifest"] = link.relative_to(tmp_path).as_posix()
    target["checkpoint_manifest_sha256"] = _sha(link)
    index["index_hash"] = hash_json({k: v for k, v in index.items() if k != "index_hash"})
    errors = validate_lineage_index(tmp_path, index, expected_stage="PLANNER")
    assert any("must use canonical P07 artifact" in error for error in errors)
