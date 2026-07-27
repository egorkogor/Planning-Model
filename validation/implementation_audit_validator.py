from __future__ import annotations

from validation.role_validator import canonical_identity_hash

REQUIRED = {
    "DOMAIN_ORACLE_CORRECTNESS",
    "GENERATOR_REPRODUCIBILITY",
    "RUNTIME_CHECKER_COVERAGE",
    "DATASET_SPLIT_AND_LEAKAGE",
    "EVALUATOR_HARNESS",
    "MODEL_LOADING_AND_RUNTIME_PINS",
    "RESULT_PERSISTENCE_AND_HASHING",
    "CLEAN_PREFLIGHT_REPRODUCTION",
    "ANALYSIS_INPUT_BUILDERS",
    "SEMANTIC_RESOLVER_AND_PROTOTYPE_BUILDER",
    "CONTROL_CERTIFICATION_ENGINE",
    "SEALER_AND_DISPATCH_BOUNDARY",
}

def validate_implementation_audit(obj: dict, resource_plan: dict) -> list[str]:
    errors: list[str] = []
    builder = resource_plan["builder"]
    expected = resource_plan["auditor"]
    reviewer = obj["reviewer"]
    ids = [row["check_id"] for row in obj["checks"]]
    if set(ids) != REQUIRED or len(ids) != len(REQUIRED):
        errors.append("implementation audit check set incomplete or duplicated")
    if any(row["status"] != "PASS" for row in obj["checks"]) and obj["decision"] == "APPROVE":
        errors.append("cannot APPROVE with failed implementation checks")
    if reviewer.get("reviewer_type") == "SERVICE_PROCESS":
        errors.append("implementation audit requires judgment; SERVICE_PROCESS is not sufficient")
    independent = reviewer.get("reviewer_type") == "HUMAN_STATISTICIAN" or reviewer.get("model_family") != builder.get("model_family")
    if not independent:
        errors.append("implementation auditor is not judgment-independent from Builder")
    declared = "HUMAN_REVIEWER" if reviewer.get("reviewer_type") == "HUMAN_STATISTICIAN" else "DIFFERENT_MODEL_FAMILY"
    if obj.get("reviewer_independence") != declared:
        errors.append("reviewer_independence declaration mismatch")
    if reviewer != expected:
        errors.append("implementation audit reviewer differs from locked resource plan identity")
    if obj.get("builder_identity_sha256") != canonical_identity_hash(builder):
        errors.append("builder_identity_sha256 mismatch")
    if obj.get("public_key_id") != reviewer.get("public_signing_key_id"):
        errors.append("implementation audit public_key_id differs from reviewer identity")
    return errors
