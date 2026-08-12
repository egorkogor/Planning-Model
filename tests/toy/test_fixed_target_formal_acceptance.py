from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

import scripts.fixed_target_acceptance_v1_1 as v11
import scripts.fixed_target_contract as ft
import scripts.run_fixed_target_acceptance as cli
from tests.toy.test_fixed_cpu_target import valid_acceptance, valid_contract

H1 = "sha256:" + "1" * 64
H2 = "sha256:" + "2" * 64
H3 = "sha256:" + "3" * 64
COMMIT = "a" * 40


def _claims() -> dict[str, str]:
    return {field: H1 for field in ft.CLAIM_IDENTITY_FIELDS}


def _units(index: int) -> list[dict]:
    attempt_id = "sha256:" + format(index, "064x")
    result = []
    counter = 0
    for variant in ("A2", "A3", "A4"):
        for seed in (17, 29, 43):
            counter += 1
            unit = {
                "attempt_identity_sha256": attempt_id,
                "variant": variant,
                "seed": seed,
                "unit_manifest_sha256": "sha256:" + format(index * 100 + counter, "064x"),
                "checkpoint_manifest_sha256": "sha256:"
                + format(index * 1000 + counter, "064x"),
                "task_results_sha256": "sha256:" + format(index * 10000 + counter, "064x"),
            }
            unit["checkpoint_lineage_sha256"] = ft.sha256_value(
                {
                    "attempt_identity_sha256": unit["attempt_identity_sha256"],
                    "variant": unit["variant"],
                    "seed": unit["seed"],
                    "unit_manifest_sha256": unit["unit_manifest_sha256"],
                    "checkpoint_manifest_sha256": unit["checkpoint_manifest_sha256"],
                }
            )
            result.append(unit)
    return result


def _summaries() -> list[dict]:
    contract = valid_contract()
    runtime = ft.build_runtime_contract(contract)
    claims = _claims()
    common = {
        "workflow_run_id": 1234,
        "job_id": 5678,
        "workflow_sha": COMMIT,
        "execution_implementation_commit": COMMIT,
        "scientific_parent_implementation_commit": ft.HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
        "target_contract": contract,
        "target_contract_sha256": ft.target_contract_sha256(contract),
        "runtime_contract": runtime,
        "runtime_contract_sha256": ft.runtime_contract_sha256(runtime),
        "target_observation_sha256": H2,
        "source_inventory_sha256": H2,
        "execution_evidence_sha256": H3,
        "observed_optimizer_foreach": False,
        "observed_optimizer_fused": False,
        "claim_identities": claims,
        "canonical_result_identity": ft.canonical_result_identity(claims),
        "training_execution_mode": v11.FORMAL_TRAINING_MODE,
        "successful_full_evaluation": True,
        "result": "PASS",
        "execution_provenance": {
            "evaluator_version": v11.FORMAL_EVALUATOR_VERSION,
            "evaluator_source_sha256": H2,
            "execution_topology": "SHARDED_VARIANT_SEED_SUBPROCESSES",
            "scientific_policy_sha256": H1,
            "requirements_lock_sha256": H2,
            "scientific_parent_implementation_commit": ft.HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
        },
    }
    result = []
    for index in (1, 2, 3):
        value = copy.deepcopy(common)
        value.update(
            {
                "attempt_index": index,
                "attempt_identity_sha256": "sha256:" + format(index, "064x"),
                "attempt_manifest_sha256": "sha256:" + format(index + 10, "064x"),
                "formal_provenance_sha256": "sha256:" + format(index + 20, "064x"),
                "units": _units(index),
            }
        )
        result.append(value)
    return result


def _formal_acceptance() -> dict:
    return v11.build_acceptance_record(_summaries())


def _reseal(value: dict) -> None:
    value["acceptance_identity"] = ft.acceptance_identity_sha256(value)


def test_acceptance_versions_are_not_redefined() -> None:
    assert ft.FIXED_TARGET_ACCEPTANCE_VERSION == "toy-quality-fixed-target-acceptance/1.0"
    assert ft.FORMAL_FIXED_TARGET_ACCEPTANCE_VERSION == "toy-quality-fixed-target-acceptance/1.1"
    assert v11.FORMAL_ACCEPTANCE_VERSION == ft.FORMAL_FIXED_TARGET_ACCEPTANCE_VERSION


def test_v11_envelope_uses_one_orchestration_identity_and_attempt_level_independence() -> None:
    acceptance = _formal_acceptance()
    ft._validate_schema(
        "fixed_target_acceptance_v1_1.schema.json",
        acceptance,
        "FIXED_TARGET_FORMAL_ACCEPTANCE_SCHEMA_INVALID",
    )
    assert acceptance["workflow_run_id"] == 1234
    assert acceptance["job_id"] == 5678
    assert acceptance["workflow_sha"] == COMMIT
    assert acceptance["execution_implementation_commit"] == COMMIT
    assert [item["attempt_index"] for item in acceptance["attempts"]] == [1, 2, 3]
    assert len({item["attempt_identity_sha256"] for item in acceptance["attempts"]}) == 3
    assert len(
        {
            unit["unit_manifest_sha256"]
            for attempt in acceptance["attempts"]
            for unit in attempt["units"]
        }
    ) == 27


def test_legacy_1_0_cannot_be_resealed_promoted_to_1_1() -> None:
    legacy = valid_acceptance(accepted=False)
    legacy["acceptance_version"] = v11.FORMAL_ACCEPTANCE_VERSION
    _reseal(legacy)
    with pytest.raises(ValueError, match="FORMAL_ACCEPTANCE_SCHEMA_INVALID"):
        ft._validate_schema(
            "fixed_target_acceptance_v1_1.schema.json",
            legacy,
            "FIXED_TARGET_FORMAL_ACCEPTANCE_SCHEMA_INVALID",
        )


def test_formal_1_1_cannot_be_downgraded_to_legacy_1_0() -> None:
    formal = _formal_acceptance()
    formal["acceptance_version"] = ft.FIXED_TARGET_ACCEPTANCE_VERSION
    _reseal(formal)
    with pytest.raises(ValueError, match="ACCEPTANCE_SCHEMA_INVALID"):
        ft._validate_schema(
            "fixed_target_acceptance.schema.json",
            formal,
            "FIXED_TARGET_ACCEPTANCE_SCHEMA_INVALID",
        )


def test_same_workflow_job_ids_remain_rejected_by_legacy_1_0() -> None:
    legacy = valid_acceptance(accepted=False)
    for attempt in legacy["attempts"]:
        attempt["workflow_run_id"] = 1234
        attempt["job_id"] = 5678
    _reseal(legacy)
    with pytest.raises(ValueError, match="FIXED_TARGET_ACCEPTANCE_DUPLICATE_WORKFLOW_RUN"):
        ft.validate_acceptance_record(legacy)


def test_duplicate_attempt_identity_rejected() -> None:
    summaries = _summaries()
    summaries[1]["attempt_identity_sha256"] = summaries[0]["attempt_identity_sha256"]
    with pytest.raises(ValueError, match="FIXED_TARGET_FORMAL_ATTEMPT_REUSE"):
        v11._validate_cross_attempt_independence(summaries)


def test_attempt_indices_must_be_exactly_1_2_3() -> None:
    summaries = _summaries()
    summaries[1]["attempt_index"] = 3
    with pytest.raises(ValueError, match="FIXED_TARGET_FORMAL_ATTEMPT_ORDER_MISMATCH"):
        v11._validate_cross_attempt_independence(summaries)


def test_reused_attempt_artifact_manifest_rejected() -> None:
    summaries = _summaries()
    summaries[1]["attempt_manifest_sha256"] = summaries[0]["attempt_manifest_sha256"]
    with pytest.raises(ValueError, match="FIXED_TARGET_FORMAL_ATTEMPT_ARTIFACT_REUSE"):
        v11._validate_cross_attempt_independence(summaries)


def test_any_reused_unit_identity_between_attempts_rejected() -> None:
    summaries = _summaries()
    summaries[1]["units"][0]["unit_manifest_sha256"] = summaries[0]["units"][0][
        "unit_manifest_sha256"
    ]
    with pytest.raises(ValueError, match="FIXED_TARGET_FORMAL_CROSS_ATTEMPT_UNIT_REUSE"):
        v11._validate_cross_attempt_independence(summaries)


def test_provenance_substitution_breaks_attempt_binding() -> None:
    summaries = _summaries()
    summaries[1]["units"] = copy.deepcopy(summaries[0]["units"])
    with pytest.raises(ValueError, match="FORMAL_UNIT_ATTEMPT_BINDING|CROSS_ATTEMPT_UNIT_REUSE"):
        v11._validate_cross_attempt_independence(summaries)


def test_exact_claim_equality_required() -> None:
    summaries = _summaries()
    summaries[2]["claim_identities"]["replay_hash"] = H3
    with pytest.raises(ValueError, match="FIXED_TARGET_FORMAL_CROSS_ATTEMPT_CLAIM_MISMATCH"):
        v11._validate_cross_attempt_independence(summaries)


def test_workflow_sha_must_equal_execution_commit() -> None:
    summaries = _summaries()
    for summary in summaries:
        summary["workflow_sha"] = "b" * 40
    with pytest.raises(ValueError, match="FIXED_TARGET_WORKFLOW_IMPLEMENTATION_MISMATCH"):
        v11.build_acceptance_record(summaries)


def test_missing_formal_provenance_is_structurally_rejected(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt-1"
    attempt.mkdir()
    for name in ("attempt_manifest.json", "preflight.json", "execution-evidence.json"):
        (attempt / name).write_text("{}", encoding="utf-8")
    (attempt / "evaluation").mkdir()
    with pytest.raises(ValueError, match="FIXED_TARGET_FORMAL_ATTEMPT_TOP_LEVEL_COVERAGE_MISMATCH"):
        v11.derive_formal_attempt_summary(attempt, 1)


def test_authoritative_v11_dispatch_has_no_parallel_final_validator(monkeypatch, tmp_path) -> None:
    acceptance = {"acceptance_version": v11.FORMAL_ACCEPTANCE_VERSION}
    (tmp_path / "acceptance.json").write_text(json.dumps(acceptance), encoding="utf-8")
    expected = {"valid": True, "accepted": True, "status": "FIXED_TARGET_ACCEPTED"}
    calls = []

    def fake(root, supplied):
        calls.append((Path(root), supplied))
        return expected

    monkeypatch.setattr(v11, "_validate_formal_acceptance_bundle_v1_1", fake)
    assert ft.validate_acceptance_bundle(tmp_path) == expected
    assert calls == [(tmp_path, acceptance)]
    assert not hasattr(cli, "validate_formal_acceptance_bundle")
    assert not hasattr(v11, "validate_formal_acceptance_bundle")


def test_validate_bundle_cli_uses_authoritative_validator_only(
    monkeypatch, tmp_path, capsys
) -> None:
    expected = {"valid": True, "accepted": True, "status": "FIXED_TARGET_ACCEPTED"}
    calls = []

    def fake(root):
        calls.append(Path(root))
        return expected

    monkeypatch.setattr(ft, "validate_acceptance_bundle", fake)
    assert cli.command_validate_bundle(Namespace(bundle_root=tmp_path)) == 0
    assert calls == [tmp_path]
    assert json.loads(capsys.readouterr().out) == expected


def test_authoritative_pass_cannot_fail_final_gate(monkeypatch, tmp_path) -> None:
    calls = []

    def fake(root):
        calls.append(Path(root))
        return {"valid": True, "accepted": True, "status": "FIXED_TARGET_ACCEPTED"}

    monkeypatch.setattr(ft, "validate_acceptance_bundle", fake)
    assert cli.command_final_gate(Namespace(bundle_root=tmp_path)) == 0
    assert calls == [tmp_path]


def test_final_gate_cannot_pass_without_authoritative_pass(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        ft,
        "validate_acceptance_bundle",
        lambda root: {"valid": True, "accepted": False, "status": "TARGET_PROVISIONED"},
    )
    with pytest.raises(ValueError, match="FIXED_TARGET_FINAL_GATE_NOT_ACCEPTED"):
        cli.command_final_gate(Namespace(bundle_root=tmp_path))


def test_v11_builder_deep_copies_claims_and_units() -> None:
    summaries = _summaries()
    acceptance = v11.build_acceptance_record(summaries)
    acceptance["attempts"][0]["claim_identities"]["replay_hash"] = H3
    acceptance["attempts"][0]["units"][0]["unit_manifest_sha256"] = H3
    assert summaries[0]["claim_identities"]["replay_hash"] == H1
    assert summaries[0]["units"][0]["unit_manifest_sha256"] != H3


def test_formal_source_closure_contains_authoritative_transitive_sources() -> None:
    required = {
        "scripts/fixed_target_acceptance_v1_1.py",
        "planner_toy/schemas/fixed_target_acceptance_v1_1.schema.json",
        "planner_toy/schemas/fixed_target_formal_provenance.schema.json",
    }
    assert required.issubset(ft._SHARDED_SOURCE_ADDITIONS)


def test_workflow_passes_versioned_workflow_sha_and_one_authoritative_final_gate() -> None:
    workflow_path = Path(__file__).parents[2] / ".github/workflows/fixed-target-acceptance.yml"
    workflow = workflow_path.read_text()
    assert '--workflow-sha "$FIXED_TARGET_IMPLEMENTATION_SHA"' in workflow
    assert "validate-bundle" in workflow
    assert "final-gate" in workflow
    assert "validate_formal_acceptance_bundle" not in workflow
