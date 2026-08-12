from __future__ import annotations

import json
from pathlib import Path

import yaml

import scripts.fixed_target_acceptance_v1_1 as v11
import scripts.fixed_target_contract as ft
from scripts.fixed_target_contract import validate_acceptance_record
from tests.toy.test_fixed_cpu_target import valid_contract

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


def test_fixed_target_workflow_enforces_formal_acceptance_contract() -> None:
    path = ROOT / ".github/workflows/fixed-target-acceptance.yml"
    text = path.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)

    assert set(workflow["on"]) == {"workflow_dispatch"}
    dispatch = workflow["on"]["workflow_dispatch"]
    assert dispatch["inputs"]["target_contract_path"]["required"] == "true"
    assert (
        dispatch["inputs"]["target_contract_path"]["default"]
        == "configs/fixed-cpu-target-1.0.json"
    )

    job = workflow["jobs"]["fixed-target-acceptance"]
    assert set(job["runs-on"]) == {
        "self-hosted",
        "linux",
        "x64",
        "planning-model-canonical-cpu-v1",
    }
    assert len(job["runs-on"]) == 4

    checkout = next(step for step in job["steps"] if step.get("uses") == "actions/checkout@v4")
    assert checkout["with"]["persist-credentials"] == "false"

    assert "git merge-base --is-ancestor" in text
    assert "origin/main" in text
    assert "${{ github.workflow_sha }}" in text
    assert 'test "$WORKFLOW_SHA" = "$REQUESTED_SHA"' in text
    assert "${{ github.run_attempt }}" in text
    assert "FIXED_TARGET_WORKFLOW_RERUN_FORBIDDEN" in text
    assert "pip install --upgrade pip" not in text
    assert "pip install " not in text
    assert 'git cat-file -e "HEAD:${CONTRACT_PATH}"' in text
    assert "--execution-context formal-fixed-target" in text
    assert text.count("for attempt_index in 1 2 3; do") == 1
    assert "validate-bundle" in text
    assert "final-gate" in text
    assert "FIXED_TARGET_ACCEPTANCE_EXECUTION_NOT_ENABLED" not in text
    assert "exit 78" not in text


def test_formal_target_requires_exact_canonical_runner_labels() -> None:
    contract = valid_contract()
    v11._validate_formal_target_contract(contract)

    mutated = json.loads(json.dumps(contract))
    mutated["required_runner_labels"].append("planning-model-machine-foo")
    mutated["required_runner_labels"].sort()

    # Preserve legacy target-contract/1.0 semantics while rejecting the broader formal claim.
    ft.validate_target_contract(mutated)
    assert ft.target_contract_sha256(mutated).startswith("sha256:")
    try:
        v11._validate_formal_target_contract(mutated)
    except ValueError as error:
        assert str(error) == "FIXED_TARGET_FORMAL_RUNNER_LABELS_MISMATCH"
    else:
        raise AssertionError("formal target accepted an unenforced runner label")


def test_formal_claim_bearing_upload_requires_successful_final_gate() -> None:
    path = ROOT / ".github/workflows/fixed-target-acceptance.yml"
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["fixed-target-acceptance"]["steps"]
    names = [step["name"] for step in steps]

    final_index = names.index("Final runtime 1.1 acceptance gate")
    upload_index = names.index("Upload formal evidence bundle")
    cleanup_index = names.index("Cleanup claim-bearing workspace")
    assert final_index < upload_index < cleanup_index

    final_step = steps[final_index]
    assert "final-gate" in final_step["run"]
    assert "FIXED_TARGET_RUNNER_IMAGE_ID_DRIFT:final" in final_step["run"]
    assert "FIXED_TARGET_IMPLEMENTATION_DIRTY_TREE_AT_FINAL_GATE" in final_step["run"]

    upload = steps[upload_index]
    assert upload.get("if") != "always()"
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"]["name"] == "fixed-target-runtime-1-1-${{ github.run_id }}"
    assert upload["with"]["path"] == ".fixed-target-formal/bundle"
