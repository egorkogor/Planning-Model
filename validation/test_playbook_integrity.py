from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PHASES = [f"P{i:02d}" for i in range(21)]
EXPECTED_GATES = [
    "G00_SCOPE",
    "G01_TRUST_AND_RESOURCES",
    "G06_STATISTICAL_IMPLEMENTATION_AUDIT",
    "G07_PLANNER_CONFIRMATORY_FREEZE",
    "G12_STAGE1A_CONFIRMATORY_FREEZE",
    "G16_STAGE1B_CONFIRMATORY_FREEZE",
    "G20_FINAL_ACCEPTANCE",
]


def load_yaml(rel: str):
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_phase_registry_prompt_bijection_and_noncyclic_manual_checks() -> None:
    registry = load_yaml("docs/operator/phase_registry_v1.yaml")
    phases = registry["phases"]
    assert [p["phase_id"] for p in phases] == EXPECTED_PHASES
    prompt_files = sorted((ROOT / "prompts").glob("P??_*.md"))
    assert len(prompt_files) == 21
    assert [p.name[:3] for p in prompt_files] == EXPECTED_PHASES
    for phase, prompt in zip(phases, prompt_files):
        text = prompt.read_text(encoding="utf-8")
        assert phase["phase_id"] in text
        assert "## Действия" in text
        assert "## Обязательные результаты фазы" in text
        assert "## Проверки до gate" in text
        assert "## Outcomes и переходы" in text
        assert "## Исполнительские правила" in text
        assert f"**Execution role:** `{phase['execution_role']}`" in text
        assert phase["max_internal_fix_attempts"] == 2
        assert f"reports/phase-{phase['phase_id']}.json" in phase["required_outputs"]
        if phase["manual_gate_id"]:
            assert phase["pre_gate_checks"]
            assert phase["post_gate_checks"]
            assert all("DecisionRecord" not in x["description"] and "решение оператора" not in x["description"] for x in phase["pre_gate_checks"])


def test_state_machine_total_for_declared_outcomes() -> None:
    registry = load_yaml("docs/operator/phase_registry_v1.yaml")
    sm = load_yaml("docs/operator/phase_state_machine_v1.yaml")
    assert sm["initial_phase"] == "P00"
    for phase in registry["phases"]:
        pid = phase["phase_id"]
        assert pid in sm["transitions"]
        assert set(sm["transitions"][pid]) == set(phase["allowed_outcomes"])
        for transition in sm["transitions"][pid].values():
            assert sum(k in transition for k in ("next", "repeat", "terminal")) == 1
            if "next" in transition:
                assert transition["next"] in EXPECTED_PHASES
            if "repeat" in transition:
                assert transition["repeat"] in EXPECTED_PHASES
                assert transition["max_resubmissions"] == 1
    assert sm["global_rules"]["do_not_advance_by_numeric_phase_order"] is True


def test_manual_gates_identical_across_contracts_and_schema() -> None:
    agent = load_yaml("docs/operator/agent_execution_contract_v1.yaml")
    registry = load_yaml("docs/operator/phase_registry_v1.yaml")
    registry_gates = [p["manual_gate_id"] for p in registry["phases"] if p["manual_gate_id"]]
    assert registry_gates == EXPECTED_GATES
    assert agent["manual_approval_gates"] == EXPECTED_GATES
    decision = json.loads((ROOT / "docs/schemas/decision_record.schema.json").read_text())
    assert decision["properties"]["gate_id"]["enum"] == EXPECTED_GATES


def test_master_prompt_points_to_current_sources_and_roles() -> None:
    text = (ROOT / "prompts/00_MASTER_ORCHESTRATOR.md").read_text(encoding="utf-8")
    for needle in (
        "Implementation_Spec_RU_v1.17.md",
        "Operator_Runbook_v2.17_RU.md",
        "phase_state_machine_v1.yaml",
        "contract_lock_v1.yaml",
        "report_registry_v1.yaml",
        "Data Sealer",
        "Evaluation Runner",
        "Audit Agent",
        "WAITING_APPROVAL",
    ):
        assert needle in text


def test_report_registry_covers_required_json_outputs() -> None:
    registry = load_yaml("docs/operator/phase_registry_v1.yaml")
    report_registry = load_yaml("docs/operator/report_registry_v1.yaml")
    patterns = [r.get("path_pattern", "") for r in report_registry["rules"]]
    schemas = {r.get("schema") or r.get("schema_per_line") for r in report_registry["rules"]}
    for schema in schemas:
        assert (ROOT / schema).exists()
    assert "reports/phase-P??.json" in patterns
    assert "RUN_STATUS.json" in patterns
    assert "artifacts/index.json" in patterns
    import fnmatch
    for phase in registry["phases"]:
        all_outputs=list(phase["required_outputs"])+list(phase.get("pre_gate_required_outputs",[]) or [])+list(phase.get("post_gate_required_outputs",[]) or [])
        for path in all_outputs:
            if path.endswith(".json"):
                assert any(fnmatch.fnmatch(path, pattern) for pattern in patterns), f"unregistered required JSON: {path}"


def test_current_normative_tree_has_no_stale_versions_or_docx() -> None:
    paths = list((ROOT / "docs").rglob("*")) + list((ROOT / "prompts").rglob("*"))
    assert not [p for p in paths if p.suffix.lower() == ".docx"]
    for path in paths:
        if not path.is_file() or "CHANGELOG" in path.name:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "work-planner/1.5" not in text
        assert "Runbook v2.5" not in text
        assert "P00–P16" not in text
        assert "**Версия:** 1.6" not in text
        assert "**Версия:** 2.6" not in text
        assert "v1.7/v2.7" not in text


def test_phase_prompts_are_exact_renderings_of_registry() -> None:
    from scripts.generate_phase_prompts import FILES, render
    registry=load_yaml("docs/operator/phase_registry_v1.yaml")
    sm=load_yaml("docs/operator/phase_state_machine_v1.yaml")
    for phase in registry["phases"]:
        expected=render(phase,sm["transitions"])
        actual=(ROOT/"prompts"/FILES[phase["phase_id"]]).read_text(encoding="utf-8")
        assert actual==expected


def test_every_pre_and_execution_check_has_locked_machine_verifier() -> None:
    registry=yaml.safe_load((ROOT/'docs/operator/phase_registry_v1.yaml').read_text(encoding='utf-8'))
    contract=yaml.safe_load((ROOT/'docs/operator/phase_check_contract_v1.yaml').read_text(encoding='utf-8'))
    core=set(contract['core_checks'])
    for phase in registry['phases']:
        for section in ('pre_gate_checks','execution_checks'):
            for check in phase.get(section,[]) or []:
                assert check.get('verifier'), (phase['phase_id'],check['check_id'])
                assert 'validation/phase_check_runner.py' in check['verifier']
                if check['check_id'] not in core:
                    assert contract['runtime_checker']['module']=='src.validation.phase_checks'


def test_post_gate_checks_are_recomputed_by_gate_verifier() -> None:
    text=(ROOT/'validation/verify_gate.py').read_text(encoding='utf-8')
    for token in ('manual gate PASS requires DecisionRecord','decision target hash does not match locked approval target','reports/gate-ledger.jsonl','reports/decision-log.jsonl'):
        assert token in text

def test_state_machine_flags_are_persisted_and_machine_checked():
    import json
    sm=load_yaml('docs/operator/phase_state_machine_v1.yaml')
    assert sm['transitions']['P09']['GO_PLANNER_STAGE1B_ELIGIBLE']['set_flags']['stage1b_eligible'] is True
    assert sm['transitions']['P14']['GO_INTERFACE_STAGE1B_ELIGIBLE']['requires_flags']['stage1b_eligible'] is True
    assert sm['transitions']['P14']['GO_INTERFACE_STAGE1B_ELIGIBLE']['set_flags']['interface_go'] is True
    schema=json.loads((ROOT/'docs/schemas/run_status.schema.json').read_text())
    assert 'flags' in schema['required']
    text=(ROOT/'validation/verify_gate.py').read_text()
    assert 'state-machine required flag' in text and 'state-machine set flag' in text
