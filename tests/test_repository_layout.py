from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_autonomous_prompt_set_is_complete() -> None:
    assert (ROOT / "prompts" / "00_MASTER_ORCHESTRATOR.md").is_file()
    phase_prompts = sorted((ROOT / "prompts").glob("P??_*.md"))
    assert len(phase_prompts) == 21
    assert {p.name[:3] for p in phase_prompts} == {f"P{i:02d}" for i in range(21)}


def test_phase_registry_has_exact_order_and_gates() -> None:
    registry = yaml.safe_load((ROOT / "docs/operator/phase_registry_v1.yaml").read_text(encoding="utf-8"))
    phases = registry["phases"]
    assert [p["phase_id"] for p in phases] == [f"P{i:02d}" for i in range(21)]
    gates = [p["manual_gate_id"] for p in phases if p["manual_gate_id"]]
    assert gates == [
        "G00_SCOPE",
        "G01_TRUST_AND_RESOURCES",
        "G06_STATISTICAL_IMPLEMENTATION_AUDIT",
        "G07_PLANNER_CONFIRMATORY_FREEZE",
        "G12_STAGE1A_CONFIRMATORY_FREEZE",
        "G16_STAGE1B_CONFIRMATORY_FREEZE",
        "G20_FINAL_ACCEPTANCE",
    ]


def test_current_markdown_specs_and_contracts_exist() -> None:
    assert (ROOT / "docs/Planner_MVP_MicroModel_Implementation_Spec_RU_v1.14.md").is_file()
    assert (ROOT / "docs/Planner_LLM_Stage1_Operator_Runbook_v2.14_RU.md").is_file()
    assert (ROOT / "docs/operator/AUTONOMOUS_EXECUTION_PLAYBOOK_RU.md").is_file()
    assert (ROOT / "docs/prompt/stage1_prompt_v1.yaml").is_file()
    assert (ROOT / "docs/semantic/semantic_target_v1.yaml").is_file()
    assert not list((ROOT / "docs").rglob("*.docx"))


def test_monitoring_and_output_directories_exist() -> None:
    assert (ROOT / "RUN_STATUS.md").is_file()
    for path in (
        "reports",
        "artifacts",
        "locks",
        "data/manifests",
        "semantic_bank",
        "checkpoints",
        "controls",
        "freezes",
        "results",
        "release",
    ):
        assert (ROOT / path).is_dir()
