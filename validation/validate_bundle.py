from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "schemas"


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--skip-nested-pytest", action="store_true")
    args=ap.parse_args()
    schemas = []
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(obj)
        assert obj["$id"].startswith("https://ml-brain.local/schemas/work-planner/v1.16/")
        schemas.append(path)

    yamls = []
    for path in sorted((ROOT / "docs").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data is not None, f"empty YAML: {path}"
        yamls.append(path)

    required = [
        ROOT / "docs/Planner_MVP_MicroModel_Implementation_Spec_RU_v1.16.md",
        ROOT / "docs/Planner_LLM_Stage1_Operator_Runbook_v2.16_RU.md",
        ROOT / "docs/operator/AUTONOMOUS_EXECUTION_PLAYBOOK_RU.md",
        ROOT / "docs/operator/phase_state_machine_v1.yaml",
        ROOT / "docs/operator/contract_lock_v1.yaml",
        ROOT / "docs/domain/intent_labeler_v1.py",
        ROOT / "prompts/00_MASTER_ORCHESTRATOR.md",
    ]
    for path in required:
        assert path.exists(), f"missing normative artifact: {path}"

    prompt = yaml.safe_load((ROOT / "docs/prompt/stage1_prompt_v1.yaml").read_text(encoding="utf-8"))
    assert prompt["chat_rendering"]["right_padding"] == "forbidden"
    assert prompt["chat_rendering"]["generation"]["tokenizer_padding_side_for_batches"] == "left"
    assert prompt["guidance_blocks"]["attended_token_budget"] == 32

    phases = yaml.safe_load((ROOT / "docs/operator/phase_registry_v1.yaml").read_text(encoding="utf-8"))["phases"]
    assert [p["phase_id"] for p in phases] == [f"P{i:02d}" for i in range(21)]

    if not args.skip_nested_pytest:
        env=dict(__import__("os").environ); env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"]="1"
        subprocess.run([sys.executable, "-m", "pytest", "-q", "validation"], cwd=ROOT, check=True, env=env)
    print(f"PASS: {len(schemas)} schemas valid; {len(yamls)} YAML contracts parsed" + ("; pytest skipped" if args.skip_nested_pytest else "; pytest passed"))


if __name__ == "__main__":
    main()
