from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from validation.hashing import decision_record_hash
from validation.confirmatory_lineage_validator import (
    result_artifact_map,
    validate_sealed_count_commitment,
)
from validation.operator_decision_validator import (
    sign_record,
    signing_bytes,
    verify_decision_history,
    verify_operator_decision,
    verify_run_decision_history,
)

ROOT = Path(__file__).resolve().parents[1]


def _keys(base: Path) -> tuple[Path, Path, str]:
    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_raw = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    base.mkdir(parents=True, exist_ok=True)
    private_path = base / "operator-private.b64"
    public_path = base / "operator-public.b64"
    private_path.write_bytes(base64.b64encode(private_raw))
    public_path.write_bytes(base64.b64encode(public_raw))
    return private_path, public_path, "sha256:" + hashlib.sha256(public_raw).hexdigest()


def _unsigned(decision_id: str, gate_id: str, outcome: str, timestamp: str) -> dict:
    return {
        "schema_version": "work-planner-agent/1.3",
        "decision_id": decision_id,
        "run_id": "run-security-test",
        "gate_id": gate_id,
        "timestamp": timestamp,
        "decision": "APPROVE",
        "phase_outcome": outcome,
        "target_artifact_hash": "sha256:" + "a" * 64,
        "evidence_hashes": ["sha256:" + "a" * 64],
        "operator_note": "",
        "resubmission_index": 0,
        "previous_decision_hash": None,
        "decision_hash": "sha256:" + "0" * 64,
        "operator_key_id": "operator-root",
        "operator_public_key_sha256": "sha256:" + "0" * 64,
        "signature_algorithm": "ed25519",
        "signature": "",
    }


def _write_input(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_signed_decision_chain_detects_deletion_and_allows_historical_gate_validation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "decisions").mkdir(parents=True)
    private_path, public_path, fingerprint = _keys(tmp_path / "keys")

    first_input = _write_input(
        tmp_path / "first.json",
        _unsigned("D-0001", "G00_SCOPE", "APPROVE_SCOPE", "2026-07-27T10:00:00Z"),
    )
    first_path = repo / "decisions/D-0001.json"
    first = sign_record(first_input, first_path, private_path, public_path, root=repo)
    assert not verify_operator_decision(first, public_path, require_trust_lock=False, root=repo)
    assert not verify_decision_history(first, public_path, require_trust_lock=False, root=repo)

    (repo / "locks").mkdir()
    (repo / "locks/trust-topology.lock.json").write_text(
        json.dumps(
            {
                "run_id": "run-security-test",
                "operator_key_id": "operator-root",
                "operator_public_key_sha256": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    second_input = _write_input(
        tmp_path / "second.json",
        _unsigned(
            "D-0002",
            "G01_TRUST_AND_RESOURCES",
            "APPROVE_TRUST_AND_RESOURCES",
            "2026-07-27T10:01:00Z",
        ),
    )
    second_path = repo / "decisions/D-0002.json"
    second = sign_record(second_input, second_path, private_path, public_path, root=repo)

    # A historical approval remains valid after a later gate decision.
    assert not verify_decision_history(first, public_path, require_trust_lock=True, root=repo)
    assert not verify_decision_history(second, public_path, require_trust_lock=True, root=repo)

    first_path.unlink()
    errors = verify_decision_history(second, public_path, require_trust_lock=True, root=repo)
    assert any("previous_decision_hash" in error for error in errors)


def test_operator_signer_refuses_fake_history_and_external_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "decisions").mkdir(parents=True)
    private_path, public_path, _ = _keys(tmp_path / "keys")
    first_input = _write_input(
        tmp_path / "first.json",
        _unsigned("D-0001", "G00_SCOPE", "APPROVE_SCOPE", "2026-07-27T10:00:00Z"),
    )
    with pytest.raises(ValueError, match="decisions/D-NNNN"):
        sign_record(first_input, tmp_path / "outside.json", private_path, public_path, root=repo)

    first_path = repo / "decisions/D-0001.json"
    first = sign_record(first_input, first_path, private_path, public_path, root=repo)
    first["operator_note"] = "tampered"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_input = _write_input(
        tmp_path / "second.json",
        _unsigned("D-0002", "G01_TRUST_AND_RESOURCES", "APPROVE_TRUST_AND_RESOURCES", "2026-07-27T10:01:00Z"),
    )
    with pytest.raises(ValueError, match="invalid signed DecisionRecord history"):
        sign_record(second_input, repo / "decisions/D-0002.json", private_path, public_path, root=repo)



def test_operator_signer_is_strictly_append_only_and_history_is_schema_validated(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "decisions").mkdir(parents=True)
    private_path, public_path, _ = _keys(tmp_path / "keys")
    first_input = _write_input(
        tmp_path / "first.json",
        _unsigned("D-0001", "G00_SCOPE", "APPROVE_SCOPE", "2026-07-27T10:00:00Z"),
    )
    first_path = repo / "decisions/D-0001.json"
    first = sign_record(first_input, first_path, private_path, public_path, root=repo)
    with pytest.raises(ValueError, match="append-only"):
        sign_record(first_input, first_path, private_path, public_path, root=repo)

    # A cryptographically valid record with an unknown field must still fail the full schema.
    first["unexpected_field"] = "schema bypass probe"
    first["decision_hash"] = decision_record_hash(first)
    raw_private = base64.b64decode(private_path.read_bytes(), validate=True)
    private = Ed25519PrivateKey.from_private_bytes(raw_private)
    first["signature"] = base64.b64encode(private.sign(signing_bytes(first))).decode("ascii")
    first_path.write_text(json.dumps(first), encoding="utf-8")
    errors = verify_decision_history(first, public_path, require_trust_lock=False, root=repo)
    assert any("Additional properties are not allowed" in error for error in errors)


def test_every_phase_can_detect_changed_prior_approval_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "decisions").mkdir(parents=True)
    (repo / "artifacts").mkdir(parents=True)
    scope = repo / "artifacts/scope.md"
    scope.write_text("locked scope", encoding="utf-8")
    private_path, public_path, _ = _keys(tmp_path / "keys")
    unsigned = _unsigned("D-0001", "G00_SCOPE", "APPROVE_SCOPE", "2026-07-27T10:00:00Z")
    unsigned["target_artifact_hash"] = "sha256:" + hashlib.sha256(scope.read_bytes()).hexdigest()
    unsigned["evidence_hashes"] = [unsigned["target_artifact_hash"]]
    first_input = _write_input(tmp_path / "first.json", unsigned)
    sign_record(first_input, repo / "decisions/D-0001.json", private_path, public_path, root=repo)
    assert not verify_run_decision_history(
        "run-security-test",
        public_path,
        require_trust_lock=False,
        expected_approved_gates={"G00_SCOPE"},
        root=repo,
    )
    scope.write_text("changed after approval", encoding="utf-8")
    errors = verify_run_decision_history(
        "run-security-test",
        public_path,
        require_trust_lock=False,
        expected_approved_gates={"G00_SCOPE"},
        root=repo,
    )
    assert any("approved target changed" in error for error in errors)



def test_rejected_target_can_be_replaced_for_bounded_resubmission(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "decisions").mkdir(parents=True)
    (repo / "artifacts").mkdir(parents=True)
    scope = repo / "artifacts/scope.md"
    scope.write_text("rejected scope", encoding="utf-8")
    private_path, public_path, _ = _keys(tmp_path / "keys")
    unsigned = _unsigned("D-0001", "G00_SCOPE", "REJECT_SCOPE", "2026-07-27T10:00:00Z")
    unsigned["decision"] = "REJECT"
    unsigned["target_artifact_hash"] = "sha256:" + hashlib.sha256(scope.read_bytes()).hexdigest()
    unsigned["evidence_hashes"] = [unsigned["target_artifact_hash"]]
    first_input = _write_input(tmp_path / "first.json", unsigned)
    sign_record(first_input, repo / "decisions/D-0001.json", private_path, public_path, root=repo)
    scope.write_text("replacement candidate", encoding="utf-8")
    errors = verify_run_decision_history(
        "run-security-test",
        public_path,
        require_trust_lock=False,
        expected_approved_gates=set(),
        root=repo,
    )
    assert not any("approved target changed" in error for error in errors)

def test_required_prior_gate_cannot_be_skipped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "decisions").mkdir(parents=True)
    errors = verify_run_decision_history(
        "run-security-test",
        require_trust_lock=False,
        expected_approved_gates={"G00_SCOPE", "G01_TRUST_AND_RESOURCES"},
        root=repo,
    )
    assert any("history is empty" in error for error in errors)

def test_confirmatory_approval_outputs_are_mandatory() -> None:
    phases = yaml.safe_load((ROOT / "docs/operator/phase_registry_v1.yaml").read_text(encoding="utf-8"))["phases"]
    expected = {
        "P07": ["freezes/planner-confirmatory.approved.json", "dispatch/evaluator-planner.json"],
        "P12": ["freezes/stage1a-confirmatory.approved.json", "dispatch/evaluator-stage1a.json"],
        "P16": ["freezes/stage1b-confirmatory.approved.json", "dispatch/evaluator-stage1b.json"],
    }
    for phase_id, outputs in expected.items():
        phase = next(row for row in phases if row["phase_id"] == phase_id)
        assert phase["post_gate_required_outputs_by_outcome"]["APPROVE_FREEZE"] == outputs
        check_ids = {row["check_id"] for row in phase["post_gate_checks_by_outcome"]["APPROVE_FREEZE"]}
        assert {f"{phase_id}_post_04", f"{phase_id}_post_05", f"{phase_id}_post_06"} <= check_ids


def test_security_validators_are_in_scientific_trust_root() -> None:
    policy = yaml.safe_load((ROOT / "docs/operator/scientific_lock_v1.yaml").read_text(encoding="utf-8"))
    protected = set(policy["protected_paths"])
    required = {
        "validation/operator_decision_validator.py",
        "validation/confirmatory_lineage_validator.py",
        "validation/code_fingerprint.py",
        "docs/schemas/approved_freeze_pointer.schema.json",
        "docs/schemas/dispatch_record.schema.json",
        "docs/schemas/evaluator_result_manifest.schema.json",
        "docs/schemas/experiment_freeze.schema.json",
        "docs/schemas/sealer_manifest.schema.json",
    }
    assert required <= protected
    manifest = json.loads((ROOT / "release/BOOTSTRAP_MANIFEST.json").read_text(encoding="utf-8"))
    assert required <= set(manifest["files"])


def test_bootstrap_detects_security_validator_tampering() -> None:
    target = ROOT / "validation/operator_decision_validator.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# tamper probe\n")
        result = subprocess.run(
            [sys.executable, str(ROOT / "validation/verify_release_manifest.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        assert "operator_decision_validator.py" in result.stdout
    finally:
        target.write_bytes(original)



def test_sealed_dataset_count_cannot_be_smaller_than_locked_n() -> None:
    freeze = {
        "sample_size": 4000,
        "sealed_dataset_commitment": {
            "task_count": 3999,
            "strata_counts": {"short": 2000, "long": 1999},
        },
    }
    sealer = {"task_count": 3999, "strata_counts": {"short": 2000, "long": 1999}}
    errors = validate_sealed_count_commitment(freeze, sealer)
    assert any("sample_size" in error for error in errors)


def test_signed_evaluator_artifact_map_binds_paths_not_only_hashes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    result = root / "results/planner-confirmatory"
    result.mkdir(parents=True)
    (result / "a.jsonl").write_text("same", encoding="utf-8")
    (result / "b.jsonl").write_text("same", encoding="utf-8")
    mapping = result_artifact_map(result, root=root)
    assert set(mapping) == {
        "results/planner-confirmatory/a.jsonl",
        "results/planner-confirmatory/b.jsonl",
    }
    assert len(set(mapping.values())) == 1



def test_phase_and_decision_timestamps_are_ordered() -> None:
    from validation.verify_gate import verify_phase_timing
    report = {
        "started_at": "2026-07-27T10:00:00Z",
        "finished_at": "2026-07-27T10:02:00Z",
        "status": "PASS",
    }
    decision = {"timestamp": "2026-07-27T10:03:00Z"}
    assert any("later than phase finish" in error for error in verify_phase_timing(report, decision))
    report["finished_at"] = "2026-07-27T09:59:00Z"
    assert any("predates started_at" in error for error in verify_phase_timing(report))

def test_confirmatory_schemas_fail_closed_on_empty_or_malformed_lineage() -> None:
    schemas = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in (ROOT / "docs/schemas").glob("*.json")
    }
    dispatch = schemas["dispatch_record.schema.json"]
    evaluator = schemas["evaluator_result_manifest.schema.json"]
    freeze = schemas["experiment_freeze.schema.json"]
    sealer = schemas["sealer_manifest.schema.json"]
    assert dispatch["properties"]["dispatch_id"]["type"] == "string"
    assert dispatch["properties"]["run_id"]["pattern"] == "^run-[a-z0-9-]+$"
    assert dispatch["properties"]["input_hashes"]["minProperties"] >= 5
    assert evaluator["properties"]["raw_artifacts"]["minProperties"] == 1
    assert evaluator["properties"]["raw_artifacts"]["propertyNames"]["pattern"].startswith("^results/")
    assert freeze["properties"]["model_locks"]["minProperties"] == 1
    assert freeze["properties"]["checkpoint_hashes"]["minProperties"] == 1
    assert {"dispatch_id", "dispatch_hash"} <= set(sealer["required"])
