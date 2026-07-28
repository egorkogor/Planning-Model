from __future__ import annotations


def validate_sealer_manifest_semantics(obj: dict) -> list[str]:
    errors: list[str] = []
    selected_sha = obj.get("selected_task_manifest_sha256")
    selected_path = obj.get("selected_task_manifest_path")
    if not selected_sha or not selected_path:
        errors.append("sealer manifest must bind selected-task manifest path and sha256")
    if obj.get("stage") != "STAGE1B":
        return errors
    cert = obj.get("control_certification")
    if not isinstance(cert, dict):
        return ["Stage1B sealer manifest requires task-only control_certification"]
    candidate = int(cert.get("candidate_task_count", 0))
    eligible = int(cert.get("eligible_task_count", 0))
    selected = int(cert.get("selected_task_count", 0))
    task_count = int(obj.get("task_count", 0))
    if not (candidate >= eligible >= selected == task_count >= 1):
        errors.append("Stage1B certification counts must satisfy candidate >= eligible >= selected == task_count >= 1")
    if cert.get("selection_basis") != "TASK_AND_DOMAIN_METADATA_ONLY":
        errors.append("Stage1B selection basis must be task/domain metadata only")
    for field in ("planner_output_used_for_selection", "llm_output_used_for_selection", "arm_outcome_used_for_selection"):
        if cert.get(field) is not False:
            errors.append(f"{field} must be false")
    if cert.get("plan_or_control_degeneracy_exclusion_count") != 0:
        errors.append("plan/control degeneracy may not exclude Stage1B tasks")
    for field in ("plan_generation_failure_policy", "control_degeneracy_policy"):
        if cert.get(field) != "RETAIN_AS_ZERO_SUCCESS_PAIRED_OUTCOME":
            errors.append(f"{field} must retain paired failures")
    if cert.get("all_selected_tasks_task_only_eligible") is not True:
        errors.append("every selected hidden Stage1B task must satisfy task-only eligibility")
    if cert.get("certification_completed_before_outcome_access") is not True:
        errors.append("hidden certification must finish before outcome access")
    contract_values = set((obj.get("contract_hashes") or {}).values())
    for field in ("eligibility_contract_sha256", "split_contract_sha256", "generator_contract_sha256"):
        if cert.get(field) not in contract_values:
            errors.append(f"{field} is not bound by contract_hashes")
    if cert.get("task_only_selection_manifest_sha256") != selected_sha:
        errors.append("control_certification is not bound to selected_task_manifest_sha256")
    return errors
