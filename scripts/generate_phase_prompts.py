from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/operator/phase_registry_v1.yaml"
STATE = ROOT / "docs/operator/phase_state_machine_v1.yaml"
FILES = {
    "P00": "P00_scope_and_protocol_start.md",
    "P01": "P01_provisioning_roles_and_budget.md",
    "P02": "P02_environment_pins_and_contract_lock.md",
    "P03": "P03_runtime_contracts_and_validation.md",
    "P04": "P04_domain_oracle_generator_intent_labeler.md",
    "P05": "P05_dataset_corpus_and_leakage.md",
    "P06": "P06_planner_models_variants.md",
    "P07": "P07_planner_development_pilot_freeze.md",
    "P08": "P08_planner_sealed_confirmatory.md",
    "P09": "P09_planner_decision_and_eligibility.md",
    "P10": "P10_frozen_llm_and_resolver.md",
    "P11": "P11_prompt_development_and_freeze.md",
    "P12": "P12_stage1a_pilot_and_freeze.md",
    "P13": "P13_stage1a_sealed_confirmatory.md",
    "P14": "P14_interface_decision.md",
    "P15": "P15_stage1b_support_and_controls.md",
    "P16": "P16_stage1b_pilot_and_freeze.md",
    "P17": "P17_stage1b_sealed_confirmatory.md",
    "P18": "P18_end_to_end_decision.md",
    "P19": "P19_independent_audit_and_reproduction.md",
    "P20": "P20_final_acceptance.md",
}
COMMON = [
    "docs/operator/agent_execution_contract_v1.yaml",
    "docs/operator/phase_state_machine_v1.yaml",
    "docs/operator/phase_registry_v1.yaml",
    "docs/operator/report_registry_v1.yaml",
    "docs/operator/self_review_loop_v1.yaml",
]


def bullets(rows):
    return "\n".join(f"- `{x}`" for x in rows) if rows else "- нет"


def checks(rows):
    return (
        "\n".join(
            f"- `{x['check_id']}` — {x['description']}"
            + (f"; verifier: `{x['verifier']}`" if x.get("verifier") else "")
            for x in rows
        )
        if rows
        else "- нет"
    )


def conditional_rows(mapping):
    if not mapping:
        return "- нет"
    out = []
    for outcome, rows in mapping.items():
        out.append(f"- `{outcome}`:")
        out.extend(
            f"  - `{row}`"
            if isinstance(row, str)
            else f"  - `{row['check_id']}` — {row['description']}"
            for row in rows
        )
    return "\n".join(out)


def render(phase, transitions):
    pid = phase["phase_id"]
    gate = phase.get("manual_gate_id") or "нет"
    role = phase["execution_role"]
    outputs = list(phase.get("required_outputs", []))
    pre = list(phase.get("pre_gate_required_outputs", []) or [])
    post = list(phase.get("post_gate_required_outputs", []) or [])
    inputs = list(phase.get("required_inputs", []) or [])
    outcome_lines = []
    for outcome in phase["allowed_outcomes"]:
        target = transitions[pid].get(outcome)
        outcome_lines.append(f"- `{outcome}` → `{target}`")
    lock_rule = (
        "- Не менять Scientific lock. Implementation-only patch разрешён только в окне и "
        "по контракту `implementation_lock_v1.yaml`."
    )
    evidence_rule = (
        "- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и "
        "self-review в phase report."
    )
    return f"""# {pid} — {phase["title"]}

**Gate:** `{gate}`
**Execution role:** `{role}`
**Approval mode:** `{phase["approval_mode"]}`

## Источники истины
{bullets(COMMON + phase.get("required_sources", []))}

## Действия
{chr(10).join(f"{i}. {x}" for i, x in enumerate(phase.get("actions", []), 1))}

## Обязательные результаты фазы
{bullets(outputs)}

## Обязательные входы фазы
{bullets(inputs)}

## Результаты до ручного gate
{bullets(pre)}

## Результаты после approval
{bullets(post)}

## Условные результаты после approval
{conditional_rows(phase.get("post_gate_required_outputs_by_outcome", {}))}

## Проверки до gate
{checks(phase.get("pre_gate_checks", []))}

## Проверки исполнения
{checks(phase.get("execution_checks", []))}

## Проверки после approval
{checks(phase.get("post_gate_checks", []))}

## Условные проверки после approval
{conditional_rows(phase.get("post_gate_checks_by_outcome", {}))}

## Outcomes и переходы
{chr(10).join(outcome_lines)}

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
{lock_rule}
{evidence_rule}
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: {phase["max_internal_fix_attempts"]}.
"""


def main():
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    sm = yaml.safe_load(STATE.read_text(encoding="utf-8"))
    for phase in reg["phases"]:
        path = ROOT / "prompts" / FILES[phase["phase_id"]]
        path.write_text(render(phase, sm["transitions"]), encoding="utf-8")
    print(f"generated {len(reg['phases'])} phase prompts")


if __name__ == "__main__":
    main()
