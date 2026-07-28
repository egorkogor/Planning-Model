from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from validation.hashing import hash_json

VARIANTS = ("A1", "A2", "A2b", "A2c", "A3", "A3r")
SEEDS = (101, 202, 303, 404, 505)

def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

def _bound_file(root: Path, rel: Any, expected: Any, label: str) -> list[str]:
    if not isinstance(rel, str) or not rel:
        return [f"{label} path missing"]
    path = (root / rel).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        return [f"{label} missing or outside repository"]
    if expected != file_digest(path):
        return [f"{label} sha256 mismatch"]
    return []

def validate_model_training_report(root: Path, obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if obj.get("report_hash") != hash_json({k: v for k, v in obj.items() if k != "report_hash"}):
        errors.append("training report self-hash mismatch")
    for rel, field in (("docs/architecture/planner_architecture_v1.yaml", "architecture_contract_sha256"),
                       ("docs/training/planner_initialization_contract_v1.yaml", "initialization_contract_sha256")):
        if obj.get(field) != file_digest(root / rel):
            errors.append(f"{field} does not bind {rel}")
    for path_field, hash_field, label in (("ordered_training_examples_path", "ordered_training_examples_sha256", "ordered training examples"),
                                         ("dormant_gradient_audit_path", "dormant_gradient_audit_sha256", "dormant gradient audit"),
                                         ("final_checkpoint_path", "final_checkpoint_sha256", "final checkpoint")):
        errors.extend(_bound_file(root, obj.get(path_field), obj.get(hash_field), label))
    if obj.get("optimizer_step") != 12000 or obj.get("checkpoint_selection") != "FINAL_STEP_ONLY":
        errors.append("final checkpoint must be optimizer step 12000 selected by FINAL_STEP_ONLY")
    return errors

def validate_final_training_matrix(root: Path, reports_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    directory = reports_dir or root / "reports/training/final"
    paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid training report {path}: {exc}")
            continue
        key = (obj.get("variant"), obj.get("seed"))
        if key in reports:
            errors.append(f"duplicate final training report for {key}")
        reports[key] = obj
        errors.extend(f"{path.name}: {e}" for e in validate_model_training_report(root, obj))
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    actual = set(reports)
    if actual != expected:
        errors.append(f"final training report matrix mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    inventories = {obj.get("parameter_inventory_sha256") for obj in reports.values()}
    if len(inventories) > 1:
        errors.append("parameter inventory hash differs across final trainable arms")
    for seed in SEEDS:
        rows = [reports[(variant, seed)] for variant in VARIANTS if (variant, seed) in reports]
        for field in ("initialization_checkpoint_sha256", "ordered_training_examples_sha256", "dataset_manifest_sha256", "config_id"):
            if len({row.get(field) for row in rows}) > 1:
                errors.append(f"{field} differs across arms for seed {seed}")
    return errors


def validate_model_audit_report(root: Path, obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if obj.get("report_hash") != hash_json({k: v for k, v in obj.items() if k != "report_hash"}):
        errors.append("model audit self-hash mismatch")
    for rel, field in (("docs/architecture/planner_architecture_v1.yaml", "architecture_contract_sha256"),
                       ("docs/training/planner_initialization_contract_v1.yaml", "initialization_contract_sha256")):
        if obj.get(field) != file_digest(root / rel):
            errors.append(f"{field} does not bind {rel}")
    expected = {"A1", "A2", "A2b", "A2c", "A3", "A3r", "A4", "A5"}
    variants = obj.get("variants", [])
    if set(variants) != expected or len(variants) != len(expected):
        errors.append("model audit variants must be the exact unique eight-variant set")
    pass_fields = ("parameter_tolerance_pass", "same_information_pass", "raw_rollout_pass", "dormant_gradient_pass")
    if obj.get("status") == "PASS" and any(obj.get(field) is not True for field in pass_fields):
        errors.append("model audit cannot PASS while a required audit flag is false")
    return errors
