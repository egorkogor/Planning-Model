from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from validation.hashing import canonical_json_bytes, hash_json
from validation.resource_plan_validator import validate_resource_plan_semantics

ROOT = Path(__file__).resolve().parents[1]


def _resource_plan() -> dict:
    obj = {
        "schema_version": "work-planner-infra/1.0",
        "run_id": "run-v121",
        "builder": {},
        "data_sealer": {},
        "evaluator": {},
        "auditor": {},
        "statistical_reviewer": {},
        "machine_checks": {
            "cpu": "PASS", "ram": "PASS", "disk": "PASS", "gpu": "NOT_APPLICABLE",
            "credentials": "PASS", "workspace": "PASS", "role_separation": "PASS",
        },
        "estimated_cost": 0,
        "currency": "USD",
        "requires_operator_budget_approval": False,
        "capacity_limits": {"maximum_gpu_seconds": 1, "maximum_storage_bytes": 1, "gpu_hour_cost": 0},
    }
    obj["plan_hash"] = hash_json(obj)
    return obj


def test_resource_plan_requires_passing_machine_checks_and_valid_self_hash():
    obj = _resource_plan()
    assert validate_resource_plan_semantics(obj) == []
    obj["machine_checks"]["disk"] = "FAIL"
    errors = validate_resource_plan_semantics(obj)
    assert "resource-plan machine check must PASS: disk" in errors
    assert "resource-plan plan_hash mismatch" in errors


def test_gpu_not_applicable_is_the_only_allowed_non_pass_machine_check():
    obj = _resource_plan()
    obj["machine_checks"]["gpu"] = "FAIL"
    obj["plan_hash"] = hash_json({k: v for k, v in obj.items() if k != "plan_hash"})
    assert validate_resource_plan_semantics(obj) == ["resource-plan GPU check must be PASS or NOT_APPLICABLE"]


def test_nfc_normalized_key_collision_is_rejected():
    with pytest.raises(ValueError, match="duplicate JSON key after NFC normalization"):
        canonical_json_bytes({"é": 1, "e\u0301": 2})


def test_static_validator_cannot_be_bypassed_by_python_optimized_mode():
    normal = subprocess.run(
        [sys.executable, "validation/validate_bundle.py", "--skip-nested-pytest"],
        cwd=ROOT, capture_output=True, text=True,
    )
    optimized = subprocess.run(
        [sys.executable, "-O", "validation/validate_bundle.py", "--skip-nested-pytest"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert normal.returncode == 0, normal.stdout + normal.stderr
    assert optimized.returncode == 0, optimized.stdout + optimized.stderr


def test_runtime_validation_modules_do_not_use_assert_statements():
    offenders = []
    for path in sorted((ROOT / "validation").glob("*.py")):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
            offenders.append(path.name)
    assert offenders == []
