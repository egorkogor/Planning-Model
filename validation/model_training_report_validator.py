from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from validation.hashing import hash_json
from validation.capacity_validator import validate_compute_profile
from validation.planner_evidence_validator import (
    FINAL_SEEDS as SEEDS,
    TRAINABLE_VARIANTS as VARIANTS,
    file_digest,
    safe_path,
    validate_checkpoint_manifest,
    validate_dormant_gradient_audit,
    validate_initialization_manifest,
    validate_model_audit_check_evidence,
    validate_ordered_examples,
    validate_parameter_inventory_manifest,
    validate_selected_config,
)


def _load_bound(root: Path, obj: Mapping[str, Any], path_field: str, hash_field: str, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        path = safe_path(root, obj.get(path_field))
        if obj.get(hash_field) != file_digest(path):
            errors.append(f"{label} sha256 mismatch")
        return json.loads(path.read_text(encoding="utf-8")), errors
    except Exception as exc:
        return None, [f"{label}: {exc}"]


def validate_model_training_report(root: Path, obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if obj.get("report_hash") != hash_json({k: v for k, v in obj.items() if k != "report_hash"}):
        errors.append("training report self-hash mismatch")
    for rel, field in (
        ("docs/architecture/planner_architecture_v1.yaml", "architecture_contract_sha256"),
        ("docs/architecture/planner_module_inventory_v1.yaml", "module_inventory_contract_sha256"),
        ("docs/training/planner_initialization_contract_v1.yaml", "initialization_contract_sha256"),
    ):
        if obj.get(field) != file_digest(root / rel):
            errors.append(f"{field} does not bind {rel}")

    config, e = _load_bound(root, obj, "selected_config_path", "selected_config_sha256", "selected config"); errors += e
    if config:
        errors += validate_selected_config(root, config)
        if obj.get("config_id") != config.get("config_id"):
            errors.append("training report config_id differs from selected config")

    dataset, e = _load_bound(root, obj, "dataset_manifest_path", "dataset_manifest_sha256", "dataset manifest"); errors += e
    if dataset and dataset.get("manifest_hash") != hash_json({k: v for k, v in dataset.items() if k != "manifest_hash"}):
        errors.append("dataset manifest self-hash mismatch")

    environment_path = root / "locks/environment.lock.json"
    if obj.get("environment_lock_path") != "locks/environment.lock.json" or not environment_path.is_file() or obj.get("environment_lock_sha256") != file_digest(environment_path):
        errors.append("training report does not bind actual environment lock")

    inventory, e = _load_bound(root, obj, "parameter_inventory_manifest_path", "parameter_inventory_manifest_sha256", "parameter inventory"); errors += e
    if inventory:
        errors += validate_parameter_inventory_manifest(root, inventory)
        if obj.get("parameter_inventory_sha256") != inventory.get("inventory_hash"):
            errors.append("training report parameter inventory content hash mismatch")

    init, e = _load_bound(root, obj, "initialization_manifest_path", "initialization_manifest_sha256", "initialization manifest"); errors += e
    if init:
        errors += validate_initialization_manifest(root, init)
        if init.get("seed") != obj.get("seed") or init.get("run_id") != obj.get("run_id"):
            errors.append("training initialization identity mismatch")

    ordered, e = _load_bound(root, obj, "ordered_training_examples_path", "ordered_training_examples_sha256", "ordered examples"); errors += e
    if ordered and dataset:
        errors += validate_ordered_examples(ordered, expected_seed=int(obj.get("seed")), dataset_sha256=obj.get("dataset_manifest_sha256"))

    audit, e = _load_bound(root, obj, "dormant_gradient_audit_path", "dormant_gradient_audit_sha256", "dormant gradient audit"); errors += e
    if audit and inventory:
        errors += validate_dormant_gradient_audit(root, audit, inventory)
        if audit.get("variant") != obj.get("variant") or audit.get("seed") != obj.get("seed") or audit.get("run_id") != obj.get("run_id"):
            errors.append("dormant gradient audit identity mismatch")

    checkpoint, e = _load_bound(root, obj, "checkpoint_manifest_path", "checkpoint_manifest_sha256", "checkpoint manifest"); errors += e
    if checkpoint:
        errors += validate_checkpoint_manifest(root, checkpoint)
        expected_kind = "FINAL_EQUAL_DATA" if obj.get("training_regime") == "FINAL_EQUAL_DATA" else "FLOPS_SENSITIVITY"
        for field in ("run_id", "variant", "seed", "optimizer_step"):
            if checkpoint.get(field) != obj.get(field):
                errors.append(f"checkpoint manifest {field} differs from training report")
        if checkpoint.get("checkpoint_kind") != expected_kind:
            errors.append("checkpoint kind differs from training regime")
        bindings = (
            ("selected_config_path", "selected_config_sha256"), ("dataset_manifest_path", "dataset_manifest_sha256"),
            ("ordered_training_examples_path", "ordered_training_examples_sha256"), ("environment_lock_path", "environment_lock_sha256"),
            ("parameter_inventory_manifest_path", "parameter_inventory_manifest_sha256"), ("initialization_manifest_path", "initialization_manifest_sha256"),
        )
        for path_field, hash_field in bindings:
            if checkpoint.get(path_field) != obj.get(path_field) or checkpoint.get(hash_field) != obj.get(hash_field):
                errors.append(f"checkpoint manifest does not bind report {path_field}")
    return errors


def _matrix(root: Path, directory: Path, expected: set[tuple[str, int]], regime: str, expected_steps: Mapping[str, int]) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    errors: list[str] = []
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
            errors.append(f"duplicate training report for {key} in {directory}")
        reports[key] = obj
        if obj.get("training_regime") != regime:
            errors.append(f"{path.name}: wrong training regime")
        if isinstance(obj.get("variant"), str) and obj.get("optimizer_step") != expected_steps.get(obj["variant"]):
            errors.append(f"{path.name}: optimizer_step differs from locked schedule")
        errors.extend(f"{path.name}: {e}" for e in validate_model_training_report(root, obj))
    actual = set(reports)
    if actual != expected:
        errors.append(f"training report matrix mismatch in {directory}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return reports, errors


def validate_final_training_matrix(root: Path, reports_dir: Path | None = None) -> list[str]:
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    reports, errors = _matrix(root, reports_dir or root / "reports/training/final", expected, "FINAL_EQUAL_DATA", {v: 12000 for v in VARIANTS})
    inventories = {obj.get("parameter_inventory_sha256") for obj in reports.values()}
    if len(inventories) > 1:
        errors.append("parameter inventory hash differs across final trainable arms")
    for seed in SEEDS:
        rows = [reports[(variant, seed)] for variant in VARIANTS if (variant, seed) in reports]
        for field in ("initialization_manifest_sha256", "ordered_training_examples_sha256", "dataset_manifest_sha256", "selected_config_sha256", "config_id", "environment_lock_sha256"):
            if len({row.get(field) for row in rows}) > 1:
                errors.append(f"{field} differs across final arms for seed {seed}")
    return errors


def validate_flops_sensitivity_matrix(root: Path, reports_dir: Path | None = None) -> list[str]:
    profile_path = root / "reports/compute-profile.json"
    if not profile_path.is_file():
        return ["compute profile missing for FLOPs-sensitivity evidence"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    errors = validate_compute_profile(profile, root)
    updates = profile.get("flops_matched_schedule", {}).get("optimizer_updates_by_variant", {})
    expected = {(variant, seed) for variant in ("A2c", "A3") for seed in SEEDS}
    reports, more = _matrix(root, reports_dir or root / "reports/training/flops-sensitivity", expected, "FLOPS_SENSITIVITY", updates)
    errors += more
    for seed in SEEDS:
        rows = [reports[(variant, seed)] for variant in ("A2c", "A3") if (variant, seed) in reports]
        for field in ("initialization_manifest_sha256", "ordered_training_examples_sha256", "dataset_manifest_sha256", "selected_config_sha256", "config_id", "environment_lock_sha256", "parameter_inventory_sha256"):
            if len({row.get(field) for row in rows}) > 1:
                errors.append(f"{field} differs across FLOPs-sensitivity arms for seed {seed}")
    return errors


def validate_all_training_evidence(root: Path) -> list[str]:
    return validate_final_training_matrix(root) + validate_flops_sensitivity_matrix(root)


MODEL_AUDIT_CHECKS = {
    "PARAMETER_TOLERANCE", "SAME_INFORMATION", "RAW_ROLLOUT", "DORMANT_GRADIENT",
    "MODEL_ARCHITECTURE_INITIALIZATION_AND_DORMANT_PARAMETERS",
}


def validate_model_audit_report(root: Path, obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if obj.get("report_hash") != hash_json({k: v for k, v in obj.items() if k != "report_hash"}):
        errors.append("model audit self-hash mismatch")
    for rel, field in (
        ("docs/architecture/planner_architecture_v1.yaml", "architecture_contract_sha256"),
        ("docs/architecture/planner_module_inventory_v1.yaml", "module_inventory_contract_sha256"),
        ("docs/training/planner_initialization_contract_v1.yaml", "initialization_contract_sha256"),
    ):
        if obj.get(field) != file_digest(root / rel):
            errors.append(f"{field} does not bind {rel}")
    expected_variants = {"A1", "A2", "A2b", "A2c", "A3", "A3r", "A4", "A5"}
    variants = obj.get("variants", [])
    if set(variants) != expected_variants or len(variants) != len(expected_variants):
        errors.append("model audit variants must be the exact unique eight-variant set")
    checks = obj.get("checks", [])
    ids = [row.get("check_id") for row in checks]
    if set(ids) != MODEL_AUDIT_CHECKS or len(ids) != len(MODEL_AUDIT_CHECKS):
        errors.append("model audit check set incomplete or duplicated")

    inventory_hashes: set[str] = set()
    for row in checks:
        check_id = row.get("check_id")
        evidence, more = _load_bound(root, row, "evidence_path", "evidence_sha256", f"model audit evidence {check_id}")
        errors += more
        if evidence:
            errors += validate_model_audit_check_evidence(root, evidence)
            expected_fields = {
                "run_id": obj.get("run_id"),
                "check_id": check_id,
                "status": row.get("status"),
                "recomputed_value": row.get("recomputed_value"),
                "expected_value": row.get("expected_value"),
            }
            for field, expected_value in expected_fields.items():
                if evidence.get(field) != expected_value:
                    errors.append(f"model audit evidence mismatch: {check_id} {field}")

            if check_id == "MODEL_ARCHITECTURE_INITIALIZATION_AND_DORMANT_PARAMETERS":
                bindings = {b.get("path"): b.get("sha256") for b in evidence.get("bindings", [])}
                required = {
                    "docs/architecture/planner_architecture_v1.yaml",
                    "docs/architecture/planner_module_inventory_v1.yaml",
                    "docs/training/planner_initialization_contract_v1.yaml",
                    "reports/model-evidence/parameter-inventory.json",
                    "reports/model-evidence/initialization/seed-17.json",
                    *{
                        f"reports/model-evidence/dormant-gradients/{variant}-seed-17.json"
                        for variant in VARIANTS
                    },
                }
                if set(bindings) != required:
                    errors.append("architecture audit evidence must bind the exact inventory, seed-17 initialization and six dormant-gradient audits")
                for rel, digest in bindings.items():
                    path = root / rel
                    if not path.is_file() or digest != file_digest(path):
                        errors.append(f"architecture audit binding invalid: {rel}")

                inv_path = root / "reports/model-evidence/parameter-inventory.json"
                inventory: dict[str, Any] = {}
                if inv_path.is_file():
                    try:
                        inventory = json.loads(inv_path.read_text(encoding="utf-8"))
                        errors += validate_parameter_inventory_manifest(root, inventory)
                        inventory_hashes.add(str(inventory.get("inventory_hash")))
                    except Exception as exc:
                        errors.append(f"architecture audit parameter inventory unreadable: {exc}")

                init_path = root / "reports/model-evidence/initialization/seed-17.json"
                if init_path.is_file():
                    try:
                        init_obj = json.loads(init_path.read_text(encoding="utf-8"))
                        errors += validate_initialization_manifest(root, init_obj)
                        if init_obj.get("run_id") != obj.get("run_id") or init_obj.get("seed") != 17:
                            errors.append("architecture audit initialization must use audit seed 17 and matching run_id")
                    except Exception as exc:
                        errors.append(f"architecture audit initialization unreadable: {exc}")

                seen_dormant: set[str] = set()
                for variant in VARIANTS:
                    audit_path = root / f"reports/model-evidence/dormant-gradients/{variant}-seed-17.json"
                    if not audit_path.is_file():
                        continue
                    try:
                        audit_obj = json.loads(audit_path.read_text(encoding="utf-8"))
                        if inventory:
                            errors += validate_dormant_gradient_audit(root, audit_obj, inventory)
                        if audit_obj.get("run_id") != obj.get("run_id") or audit_obj.get("seed") != 17 or audit_obj.get("variant") != variant:
                            errors.append(f"architecture audit dormant-gradient identity mismatch: {variant}")
                        seen_dormant.add(variant)
                    except Exception as exc:
                        errors.append(f"architecture audit dormant-gradient unreadable ({variant}): {exc}")
                if seen_dormant != set(VARIANTS):
                    errors.append("architecture audit must validate dormant gradients for all six trainable variants")

                canonical_result = {
                    "inventory_exact": True,
                    "initialization_exact": True,
                    "dormant_gradients_exact": True,
                }
                if row.get("recomputed_value") != canonical_result or row.get("expected_value") != canonical_result:
                    errors.append("architecture audit check must report the canonical three-part PASS result")

        if row.get("status") != "PASS" or row.get("recomputed_value") != row.get("expected_value"):
            if obj.get("status") == "PASS":
                errors.append("model audit cannot PASS with a failed or unequal check")
    if inventory_hashes and obj.get("parameter_inventory_sha256") not in inventory_hashes:
        errors.append("model audit parameter inventory hash differs from evidence")
    if obj.get("status") == "PASS" and errors:
        errors.append("model audit cannot PASS when bound evidence validation has errors")
    return errors
