from __future__ import annotations

from typing import Any, Mapping

from validation.hashing import hash_json

REQUIRED_PASS_CHECKS = (
    "cpu", "ram", "disk", "credentials", "workspace", "role_separation",
)


def validate_resource_plan_semantics(obj: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    checks = obj.get("machine_checks", {})
    if not isinstance(checks, Mapping):
        return ["resource-plan machine_checks must be an object"]
    for check_id in REQUIRED_PASS_CHECKS:
        if checks.get(check_id) != "PASS":
            errors.append(f"resource-plan machine check must PASS: {check_id}")
    gpu = checks.get("gpu")
    if gpu not in {"PASS", "NOT_APPLICABLE"}:
        errors.append("resource-plan GPU check must be PASS or NOT_APPLICABLE")
    submitted = obj.get("plan_hash")
    payload = {key: value for key, value in obj.items() if key != "plan_hash"}
    if submitted != hash_json(payload):
        errors.append("resource-plan plan_hash mismatch")
    return errors
