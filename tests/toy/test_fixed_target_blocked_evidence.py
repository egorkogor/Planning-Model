from __future__ import annotations

import json
from pathlib import Path

from scripts.fixed_target_contract import validate_acceptance_record

ROOT = Path(__file__).resolve().parents[2]


def test_committed_blocked_acceptance_record_is_valid() -> None:
    path = ROOT / "docs/evaluations/data/fixed-target-acceptance.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_acceptance_record(value)
    assert value["status"] == "BLOCKED_ON_FIXED_RUNNER_PROVISIONING"
    assert value["accepted"] is False
    assert value["attempts"] == []
    assert value["cross_attempt_comparison"] == {
        "mismatches": [],
        "status": "NOT_RUN",
    }
    assert value["target_contract"] is None
    assert value["runtime_contract"] is None


def test_fixed_target_workflow_is_trusted_foundation_preflight_only() -> None:
    path = ROOT / ".github/workflows/fixed-target-acceptance.yml"
    text = path.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "pull_request_target:" not in text
    assert "runs-on: [self-hosted, linux, x64, planning-model-canonical-cpu-v1]" in text
    assert "persist-credentials: false" in text
    assert "git merge-base --is-ancestor" in text
    assert "origin/main" in text
    assert "pip install --upgrade pip" not in text
    assert "pip install " not in text
    assert "FIXED_TARGET_RUNNER_LABELS" not in text
    assert "FIXED_TARGET_ACCEPTANCE_EXECUTION_NOT_ENABLED" in text
    assert "exit 78" in text
