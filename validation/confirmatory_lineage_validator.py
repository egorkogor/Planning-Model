from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT))

from validation.code_fingerprint import analysis_code_digest
from validation.hashing import approved_freeze_pointer_hash, dispatch_record_hash, experiment_freeze_hash
from validation.operator_decision_validator import verify_decision_history, verify_operator_decision
from validation.full_plan_lineage_validator import validate_lineage_index
from validation.signature_validator import verify_signed_manifest

STAGES = {
    "PLANNER_CONFIRMATORY": {
        "stage": "PLANNER",
        "gate": "G07_PLANNER_CONFIRMATORY_FREEZE",
        "candidate": "freezes/planner-confirmatory.candidate.json",
        "pointer": "freezes/planner-confirmatory.approved.json",
        "dispatch": "dispatch/evaluator-planner.json",
        "sealer": "sealed/planner-confirmatory/sealer-manifest.json",
        "sealer_dispatch": "dispatch/sealer-planner.json",
        "sample": "reports/sample-size-planner.json",
        "result_dir": "results/planner-confirmatory",
    },
    "STAGE1A_CONFIRMATORY": {
        "stage": "STAGE1A",
        "gate": "G12_STAGE1A_CONFIRMATORY_FREEZE",
        "candidate": "freezes/stage1a-confirmatory.candidate.json",
        "pointer": "freezes/stage1a-confirmatory.approved.json",
        "dispatch": "dispatch/evaluator-stage1a.json",
        "sealer": "sealed/stage1a-confirmatory/sealer-manifest.json",
        "sealer_dispatch": "dispatch/sealer-stage1a.json",
        "sample": "reports/sample-size-stage1a.json",
        "result_dir": "results/stage1a-confirmatory",
    },
    "STAGE1B_CONFIRMATORY": {
        "stage": "STAGE1B",
        "gate": "G16_STAGE1B_CONFIRMATORY_FREEZE",
        "candidate": "freezes/stage1b-confirmatory.candidate.json",
        "pointer": "freezes/stage1b-confirmatory.approved.json",
        "dispatch": "dispatch/evaluator-stage1b.json",
        "sealer": "sealed/stage1b-confirmatory/sealer-manifest.json",
        "sealer_dispatch": "dispatch/sealer-stage1b.json",
        "sample": "reports/sample-size-stage1b.json",
        "result_dir": "results/stage1b-confirmatory",
    },
}
BY_DISPATCH_STAGE = {row["stage"]: (freeze_stage, row) for freeze_stage, row in STAGES.items()}


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(rel: str) -> Path:
    path = (ROOT / rel).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise ValueError(f"path escapes repository: {rel}")
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _schemas() -> tuple[dict, Registry]:
    registry = Registry(); schemas = {}
    for path in sorted((ROOT / "docs/schemas").glob("*.json")):
        obj = load_json(path); schemas[path.name] = obj
        registry = registry.with_resource(obj["$id"], Resource.from_contents(obj))
    return schemas, registry


def schema_errors(obj: dict, name: str) -> list[str]:
    schemas, registry = _schemas()
    return [error.message for error in Draft202012Validator(schemas[name], registry=registry, format_checker=FormatChecker()).iter_errors(obj)]


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _lock_run_id(rel: str) -> str | None:
    try:
        return load_json(ROOT / rel).get("run_id")
    except Exception:
        return None


def validate_sealed_count_commitment(freeze: dict, sealer: dict) -> list[str]:
    errors: list[str] = []
    commitment = freeze.get("sealed_dataset_commitment", {})
    if commitment.get("task_count") != sealer.get("task_count"):
        errors.append("freeze task_count differs from sealer manifest")
    if commitment.get("task_count") != freeze.get("sample_size"):
        errors.append("sealed task_count must equal the locked confirmatory sample_size")
    if commitment.get("strata_counts") != sealer.get("strata_counts"):
        errors.append("freeze strata_counts differ from sealer manifest")
    try:
        if sum(int(v) for v in (commitment.get("strata_counts") or {}).values()) != commitment.get("task_count"):
            errors.append("freeze strata_counts must sum to task_count")
    except Exception:
        errors.append("freeze strata_counts must contain integer counts")
    return errors


def result_artifact_map(result_dir: Path, *, root: Path = ROOT) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in result_dir.rglob("*")
        if path.is_file() and path.name != "evaluator-result-manifest.json"
    }


def validate_experiment_freeze(obj: dict, *, expected_run_id: str | None = None) -> list[str]:
    errors = schema_errors(obj, "experiment_freeze.schema.json")
    stage = STAGES.get(obj.get("stage"))
    if stage is None:
        return errors + ["unknown confirmatory freeze stage"]
    if obj.get("freeze_hash") != experiment_freeze_hash(obj):
        errors.append("experiment freeze_hash mismatch")
    run_id = obj.get("run_id")
    if expected_run_id is not None and run_id != expected_run_id:
        errors.append("experiment freeze run_id differs from phase report")
    if obj.get("approval_gate_id") != stage["gate"]:
        errors.append("experiment freeze approval gate does not match stage")
    lock_fields = {
        "trust_topology_lock_sha256": "locks/trust-topology.lock.json",
        "scientific_lock_sha256": "locks/scientific.lock.json",
        "implementation_lock_sha256": "locks/implementation.lock.json",
        "environment_lock_sha256": "locks/environment.lock.json",
        "compute_profile_sha256": "reports/compute-profile.json",
        "capacity_preflight_sha256": "reports/preflight-final.json",
    }
    for field, rel in lock_fields.items():
        path = ROOT / rel
        if not path.is_file() or obj.get(field) != digest(path):
            errors.append(f"experiment freeze {field} does not match {rel}")
        if rel.endswith(".json") and _lock_run_id(rel) not in {None, run_id}:
            errors.append(f"experiment freeze run_id differs from {rel}")
    if obj.get("analysis_code_sha256") != analysis_code_digest():
        errors.append("experiment freeze analysis_code_sha256 mismatch")
    try:
        candidate = load_json(ROOT / "freezes/implementation-lock.candidate.json")
        if obj.get("git_commit") != candidate.get("reviewed_commit"):
            errors.append("experiment freeze git_commit differs from approved implementation commit")
        if subprocess.run(["git", "cat-file", "-e", f"{obj.get('git_commit')}^{{commit}}"], cwd=ROOT, capture_output=True).returncode:
            errors.append("experiment freeze git_commit is unavailable")
    except Exception as exc:
        errors.append(f"cannot verify experiment freeze implementation commit: {exc}")
    for rel, declared in obj.get("contract_hashes", {}).items():
        try:
            path = safe_path(rel)
            if not path.is_file() or digest(path) != declared:
                errors.append(f"experiment freeze contract hash mismatch: {rel}")
        except Exception as exc:
            errors.append(str(exc))
    for field in ("model_locks", "checkpoint_hashes"):
        for rel, declared in obj.get(field, {}).items():
            try:
                path = safe_path(rel)
                if not path.is_file() or digest(path) != declared:
                    errors.append(f"experiment freeze {field} hash mismatch: {rel}")
            except Exception as exc:
                errors.append(str(exc))
    try:
        sample = load_json(ROOT / stage["sample"])
        if sample.get("run_id") != run_id or sample.get("stage") != stage["stage"]:
            errors.append("sample-size report lineage differs from experiment freeze")
        if sample.get("selected_n") != obj.get("sample_size"):
            errors.append("experiment freeze sample_size differs from locked sample-size report")
        if sample.get("analysis_code_sha256") != obj.get("analysis_code_sha256"):
            errors.append("experiment freeze analysis code differs from sample-size report")
    except Exception as exc:
        errors.append(f"cannot verify experiment freeze sample-size lineage: {exc}")
    try:
        sealer_path = ROOT / stage["sealer"]
        sealer = load_json(sealer_path)
        errors.extend(schema_errors(sealer, "sealer_manifest.schema.json"))
        errors.extend(verify_signed_manifest(sealer, "DATA_SEALER"))
        sealer_sha = digest(sealer_path)
        if sealer.get("run_id") != run_id or sealer.get("stage") != stage["stage"]:
            errors.append("sealer manifest lineage differs from experiment freeze")
        if obj.get("sealed_dataset_sha256") != sealer.get("encrypted_blob_sha256"):
            errors.append("sealed dataset hash differs from sealer manifest")
        commitment = obj.get("sealed_dataset_commitment", {})
        errors.extend(validate_sealed_count_commitment(obj, sealer))
        if commitment.get("canonical_order_commitment_sha256") != sealer.get("canonical_order_commitment"):
            errors.append("freeze canonical order commitment differs from sealer manifest")
        if obj.get("task_manifest_sha256") != sealer_sha:
            errors.append("freeze task_manifest_sha256 must equal the signed sealer manifest file hash")
        if sealer_sha not in obj.get("dataset_manifest_hashes", []):
            errors.append("signed sealer manifest hash absent from dataset_manifest_hashes")
        sealer_dispatch_path = ROOT / stage["sealer_dispatch"]
        sealer_dispatch = load_json(sealer_dispatch_path)
        errors.extend(validate_dispatch(sealer_dispatch, expected_run_id=run_id))
        if sealer.get("dispatch_id") != sealer_dispatch.get("dispatch_id") or sealer.get("dispatch_hash") != sealer_dispatch.get("dispatch_hash"):
            errors.append("sealer manifest dispatch lineage mismatch")
        if _timestamp(sealer["created_at"]) < _timestamp(sealer_dispatch["created_at"]):
            errors.append("sealer manifest predates its dispatch")
        if stage["stage"] == "STAGE1B":
            cert = sealer.get("control_certification", {})
            required = {
                cert.get("task_only_selection_manifest_sha256"),
                cert.get("preoutcome_artifact_manifest_sha256"),
                cert.get("public_exclusion_manifest_sha256"),
            }
            if not required.issubset(set(obj.get("preoutcome_certification_hashes", []))):
                errors.append("Stage1B freeze omits signed task-only/pre-outcome certification hashes")
    except Exception as exc:
        errors.append(f"cannot verify experiment freeze sealer lineage: {exc}")
    return errors


def validate_approved_pointer(obj: dict, *, expected_run_id: str | None = None) -> list[str]:
    errors = schema_errors(obj, "approved_freeze_pointer.schema.json")
    freeze_stage = obj.get("stage")
    stage = STAGES.get(freeze_stage)
    if stage is None:
        return errors + ["unknown approved freeze pointer stage"]
    if obj.get("pointer_hash") != approved_freeze_pointer_hash(obj):
        errors.append("ApprovedFreezePointer pointer_hash mismatch")
    if expected_run_id is not None and obj.get("run_id") != expected_run_id:
        errors.append("ApprovedFreezePointer run_id differs from phase report")
    if obj.get("approval_gate_id") != stage["gate"] or obj.get("candidate_path") != stage["candidate"]:
        errors.append("ApprovedFreezePointer stage/gate/candidate mapping mismatch")
    try:
        candidate_path = safe_path(obj["candidate_path"]); candidate = load_json(candidate_path)
        if digest(candidate_path) != obj.get("candidate_sha256"):
            errors.append("ApprovedFreezePointer candidate_sha256 mismatch")
        errors.extend(validate_experiment_freeze(candidate, expected_run_id=obj.get("run_id")))
        if candidate.get("freeze_id") != obj.get("freeze_id") or candidate.get("freeze_hash") != obj.get("freeze_hash"):
            errors.append("ApprovedFreezePointer does not identify the candidate freeze")
    except Exception as exc:
        errors.append(f"cannot verify ApprovedFreezePointer candidate: {exc}")
    try:
        decision_path = safe_path(obj["decision_record_path"]); decision = load_json(decision_path)
        errors.extend(schema_errors(decision, "decision_record.schema.json"))
        errors.extend(verify_operator_decision(decision, expected_run_id=obj.get("run_id"), require_trust_lock=True, root=ROOT))
        errors.extend(verify_decision_history(decision, require_trust_lock=True, root=ROOT))
        if digest(decision_path) != obj.get("decision_record_sha256"):
            errors.append("ApprovedFreezePointer decision_record_sha256 mismatch")
        if decision.get("decision_id") != obj.get("decision_id") or decision.get("decision_hash") != obj.get("decision_hash"):
            errors.append("ApprovedFreezePointer decision identity mismatch")
        if decision.get("gate_id") != stage["gate"] or decision.get("decision") != "APPROVE" or decision.get("phase_outcome") != "APPROVE_FREEZE":
            errors.append("ApprovedFreezePointer requires an approved freeze DecisionRecord")
        if decision.get("target_artifact_hash") != obj.get("candidate_sha256"):
            errors.append("ApprovedFreezePointer decision target differs from candidate")
        if _timestamp(obj["approved_at"]) < _timestamp(decision["timestamp"]):
            errors.append("ApprovedFreezePointer predates its DecisionRecord")
    except Exception as exc:
        errors.append(f"cannot verify ApprovedFreezePointer decision: {exc}")
    return errors


def validate_dispatch(obj: dict, *, expected_run_id: str | None = None) -> list[str]:
    errors = schema_errors(obj, "dispatch_record.schema.json")
    if obj.get("dispatch_hash") != dispatch_record_hash(obj):
        errors.append("dispatch_record dispatch_hash mismatch")
    if expected_run_id is not None and obj.get("run_id") != expected_run_id:
        errors.append("dispatch run_id differs from phase report")
    if obj.get("dispatch_type") not in {"SEAL_DATASET", "RUN_CONFIRMATORY"}:
        return errors
    mapping = BY_DISPATCH_STAGE.get(obj.get("stage"))
    if mapping is None:
        return errors + ["unknown confirmatory dispatch stage"]
    freeze_stage, stage = mapping
    if obj.get("dispatch_type") == "SEAL_DATASET":
        if obj.get("source_role") != "BUILDER" or obj.get("target_role") != "DATA_SEALER":
            errors.append("sealer dispatch must be BUILDER -> DATA_SEALER")
        if obj.get("status") != "DISPATCHED":
            errors.append("sealer dispatch must be immutable DISPATCHED evidence")
        for rel in (
            "locks/trust-topology.lock.json",
            "locks/scientific.lock.json",
            "locks/implementation.lock.json",
            stage["sample"],
        ):
            path = ROOT / rel
            if not path.is_file() or obj.get("input_hashes", {}).get(rel) != digest(path):
                errors.append(f"sealer dispatch input hash mismatch: {rel}")
        return errors
    if obj.get("dispatch_type") != "RUN_CONFIRMATORY":
        return errors + ["unsupported dispatch_type for stage"]
    if obj.get("source_role") != "BUILDER" or obj.get("target_role") != "EVALUATION_RUNNER":
        errors.append("confirmatory dispatch must be BUILDER -> EVALUATION_RUNNER")
    if obj.get("status") != "DISPATCHED":
        errors.append("confirmatory dispatch must be immutable DISPATCHED evidence")
    if obj.get("approved_freeze_pointer_path") != stage["pointer"]:
        errors.append("confirmatory dispatch approved pointer path mismatch")
    try:
        pointer_path = safe_path(obj["approved_freeze_pointer_path"]); pointer = load_json(pointer_path)
        errors.extend(validate_approved_pointer(pointer, expected_run_id=obj.get("run_id")))
        if digest(pointer_path) != obj.get("approved_freeze_pointer_sha256"):
            errors.append("dispatch approved_freeze_pointer_sha256 mismatch")
        if pointer.get("pointer_hash") != obj.get("approved_freeze_pointer_hash"):
            errors.append("dispatch approved_freeze_pointer_hash mismatch")
        if pointer.get("freeze_hash") != obj.get("freeze_hash") or pointer.get("stage") != freeze_stage:
            errors.append("dispatch freeze lineage differs from approved pointer")
        required_inputs = {
            stage["pointer"]: digest(pointer_path),
            stage["candidate"]: pointer.get("candidate_sha256"),
            "locks/trust-topology.lock.json": digest(ROOT / "locks/trust-topology.lock.json"),
            "locks/scientific.lock.json": digest(ROOT / "locks/scientific.lock.json"),
            "locks/implementation.lock.json": digest(ROOT / "locks/implementation.lock.json"),
        }
        for rel, expected in required_inputs.items():
            if obj.get("input_hashes", {}).get(rel) != expected:
                errors.append(f"confirmatory dispatch input hash mismatch: {rel}")
        if _timestamp(obj["created_at"]) < _timestamp(pointer["approved_at"]):
            errors.append("confirmatory dispatch predates freeze approval")
    except Exception as exc:
        errors.append(f"cannot verify confirmatory dispatch lineage: {exc}")
    return errors


def validate_evaluator_manifest(obj: dict, report: dict) -> list[str]:
    errors = schema_errors(obj, "evaluator_result_manifest.schema.json")
    run_id = report.get("run_id")
    if obj.get("run_id") != run_id:
        errors.append("evaluator manifest run_id differs from phase report")
    mapping = BY_DISPATCH_STAGE.get(obj.get("stage"))
    if mapping is None:
        return errors + ["unknown evaluator manifest stage"]
    _, stage = mapping
    try:
        dispatch_path = ROOT / stage["dispatch"]; dispatch = load_json(dispatch_path)
        errors.extend(validate_dispatch(dispatch, expected_run_id=run_id))
        if report.get("input_hashes", {}).get(stage["dispatch"]) != digest(dispatch_path):
            errors.append("evaluator phase input_hashes do not bind the approved dispatch")
        pointer_path = ROOT / stage["pointer"]
        if report.get("input_hashes", {}).get(stage["pointer"]) != digest(pointer_path):
            errors.append("evaluator phase input_hashes do not bind the approved freeze pointer")
        if obj.get("dispatch_id") != dispatch.get("dispatch_id") or obj.get("dispatch_hash") != dispatch.get("dispatch_hash"):
            errors.append("evaluator manifest dispatch lineage mismatch")
        if obj.get("approved_freeze_pointer_hash") != dispatch.get("approved_freeze_pointer_hash"):
            errors.append("evaluator manifest approved pointer lineage mismatch")
        if obj.get("freeze_hash") != dispatch.get("freeze_hash"):
            errors.append("evaluator manifest freeze_hash differs from dispatch")
        pointer = load_json(pointer_path); freeze = load_json(ROOT / pointer["candidate_path"])
        if obj.get("git_commit") != freeze.get("git_commit"):
            errors.append("evaluator git_commit differs from approved freeze")
        if obj.get("task_count") != freeze.get("sealed_dataset_commitment", {}).get("task_count"):
            errors.append("evaluator task_count differs from approved freeze")
        if obj.get("stage") in {"PLANNER", "STAGE1B"}:
            lineage_path = ROOT / stage["result_dir"] / "lineage-index.json"
            lineage = load_json(lineage_path)
            errors.extend(f"lineage: {e}" for e in validate_lineage_index(ROOT, lineage, expected_stage=obj.get("stage")))
            lineage_task_ids = {str(row.get("task_id")) for row in lineage.get("records", [])}
            if obj.get("task_count") != len(lineage_task_ids):
                errors.append("evaluator task_count differs from exact lineage task set")
            if lineage.get("run_id") != obj.get("run_id") or lineage.get("stage") != obj.get("stage"):
                errors.append("evaluator manifest run_id/stage differs from lineage index")
            canonical_sealer_path = ROOT / stage["sealer"]
            if lineage.get("sealer_manifest") != stage["sealer"] or not canonical_sealer_path.is_file() or lineage.get("sealer_manifest_sha256") != digest(canonical_sealer_path):
                errors.append("lineage index does not bind the canonical signed sealer manifest")
        if _timestamp(obj["completed_at"]) < _timestamp(dispatch["created_at"]):
            errors.append("evaluator manifest predates its dispatch")
        result_dir = ROOT / stage["result_dir"]
        expected_raw_artifacts = result_artifact_map(result_dir, root=ROOT)
        if not expected_raw_artifacts:
            errors.append("evaluator result directory contains no signed raw artifacts")
        if obj.get("raw_artifacts") != expected_raw_artifacts:
            errors.append("evaluator raw_artifacts do not exactly cover result paths and hashes")
    except Exception as exc:
        errors.append(f"cannot verify evaluator confirmatory lineage: {exc}")
    try:
        plan = load_json(ROOT / "reports/resource-plan.json")
        env = plan["evaluator"]["environment_identity"]
        expected_env = "sha256:" + hashlib.sha256(env.encode("utf-8")).hexdigest()
        if obj.get("evaluator_environment_sha256") != expected_env:
            errors.append("evaluator environment differs from locked resource plan")
    except Exception as exc:
        errors.append(f"cannot verify evaluator environment lineage: {exc}")
    return errors
