from __future__ import annotations


def validate_sealer_manifest_semantics(obj: dict) -> list[str]:
    errors: list[str] = []
    if obj.get("stage") != "STAGE1B":
        return errors
    cert = obj.get("control_certification")
    if not isinstance(cert, dict):
        return ["Stage1B sealer manifest requires control_certification"]
    candidate = int(cert.get("candidate_task_count", 0))
    eligible = int(cert.get("eligible_task_count", 0))
    selected = int(obj.get("task_count", 0))
    if not (candidate >= eligible >= selected >= 1):
        errors.append("Stage1B certification counts must satisfy candidate >= eligible >= selected >= 1")
    if cert.get("coverage_rate") != 1.0:
        errors.append("Stage1B hidden control coverage must equal 1.0")
    if cert.get("all_selected_tasks_certified") is not True:
        errors.append("every selected hidden Stage1B task must be certified")
    if cert.get("certification_completed_before_outcome_access") is not True:
        errors.append("hidden certification must finish before outcome access")
    contract_values = set((obj.get("contract_hashes") or {}).values())
    for field in ("control_contract_sha256", "eligibility_contract_sha256", "support_contract_sha256"):
        if cert.get(field) not in contract_values:
            errors.append(f"{field} is not bound by contract_hashes")
    return errors
