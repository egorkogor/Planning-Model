from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, ValidationError

from scripts.fixed_target_contract import sha256_bytes
from scripts.fixed_target_quality_sharded import (
    EXECUTION_MODE,
    _train_runtime11,
    attempt_identity,
    checkout_commit,
    collect_runtime11_observation,
    unit_identity,
    validate_attempt,
    validate_checkout_source_inventory,
    validate_runtime11_optimizer,
    validate_unit_artifacts,
    validate_unit_manifest,
)
from tests.toy.test_fixed_cpu_target import valid_contract

SCHEMA = json.loads(
    (
        Path(__file__).parents[2] / "planner_toy/schemas/fixed_target_quality_unit.schema.json"
    ).read_text()
)
H = "sha256:" + "1" * 64


def manifest() -> dict:
    return {
        "unit_evidence_version": "toy-quality-fixed-target-sharded-unit/1.0",
        "attempt_identity_sha256": H,
        "scientific_parent_implementation_commit": ("779172c3bbca3d03552deaed6421e82fcf19a932"),
        "execution_implementation_commit": "1" * 40,
        "execution_context": "qualification-only",
        "variant": "A2",
        "seed": 17,
        "dataset_hash": H,
        "ordered_train_task_ids": ["bw-00000001", "bw-00000002", "bw-00000003"],
        "ordered_eval_task_ids": ["bw-00000004", "bw-00000005"],
        "epochs": 3,
        "updates": 9,
        "training_execution_mode": EXECUTION_MODE,
        "optimizer": {
            "class": "AdamW",
            "learning_rate": 0.0003,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "observed_foreach": False,
            "observed_fused": False,
        },
        "runtime_contract_sha256": H,
        "target_observation_sha256": H,
        "source_inventory_sha256": H,
        "observation": {},
        "checkpoint_manifest_sha256": H,
        "task_results_sha256": H,
        "unit_manifest_sha256": H,
    }


def assert_schema_rejects(field: str, value) -> None:
    candidate = manifest()
    if field.startswith("optimizer."):
        candidate["optimizer"][field.split(".")[1]] = value
    else:
        candidate[field] = value
    assert list(Draft202012Validator(SCHEMA).iter_errors(candidate))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("variant", "A1"),
        ("seed", 99),
        ("ordered_train_task_ids", []),
        ("ordered_eval_task_ids", []),
        ("epochs", 2),
        ("updates", 8),
        ("optimizer.observed_foreach", None),
        ("optimizer.observed_foreach", True),
        ("optimizer.observed_fused", None),
        ("optimizer.observed_fused", True),
        ("training_execution_mode", "REUSED_FROM_PRIOR_ATTEMPT"),
    ],
)
def test_unit_schema_rejects_contract_drift(field: str, value) -> None:
    assert_schema_rejects(field, value)


def test_unit_schema_is_closed() -> None:
    candidate = manifest()
    candidate["unexpected"] = True
    assert list(Draft202012Validator(SCHEMA).iter_errors(candidate))


def test_unit_manifest_reseal_changes_identity() -> None:
    candidate = manifest()
    first = unit_identity(candidate)
    candidate["variant"] = "A3"
    assert unit_identity(candidate) != first


def test_attempt_nonce_binds_units_but_can_be_excluded_from_numerical_claims() -> None:
    first = {"attempt_nonce": "Q1", "attempt_identity_sha256": ""}
    second = copy.deepcopy(first)
    second["attempt_nonce"] = "Q2"
    assert attempt_identity(first) != attempt_identity(second)


@pytest.mark.parametrize(
    ("foreach", "fused"), [(None, False), (True, False), (False, None), (False, True)]
)
def test_runtime11_optimizer_rejects_non_false(foreach, fused) -> None:
    with pytest.raises(ValueError, match="REQUIRED_FALSE"):
        validate_runtime11_optimizer(SimpleNamespace(defaults={"foreach": foreach, "fused": fused}))


def test_runtime11_optimizer_accepts_explicit_false() -> None:
    validate_runtime11_optimizer(SimpleNamespace(defaults={"foreach": False, "fused": False}))


def test_qualification_observation_is_actual_not_formal_target() -> None:
    forged = valid_contract()
    forged["os_version"] = "FORGED EXPECTED OS"
    observed = collect_runtime11_observation(forged, "qualification-only")
    assert observed["os_version"] != forged["os_version"]
    assert observed["runner_type"] == "codex-managed-container-qualification-only"
    assert observed["runner_image"] != valid_contract()["runner_image"]


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/fixed_target_contract.py",
        "planner_toy/schemas/fixed_target_observation.schema.json",
    ],
)
def test_checkout_inventory_detects_transitive_source_mutation(
    tmp_path, monkeypatch, relative
) -> None:
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text("original")
    inventory = {"files": [{"path": relative, "sha256": sha256_bytes(b"original")}]}
    monkeypatch.setattr("scripts.fixed_target_quality_sharded.ROOT", tmp_path)
    validate_checkout_source_inventory(inventory)
    source.write_text("mutated")
    with pytest.raises(ValueError, match="SOURCE_TREE_DRIFT"):
        validate_checkout_source_inventory(inventory)


@pytest.fixture(scope="module")
def sealed_unit(tmp_path_factory) -> Path:
    from scripts.fixed_target_quality_sharded import initialize_attempt, run_unit

    root = tmp_path_factory.mktemp("sharded-attempt")
    contract = valid_contract()
    observation = collect_runtime11_observation(contract, "qualification-only")
    initialize_attempt(
        root,
        contract,
        checkout_commit(),
        "qualification-only",
        "mutation-fixture",
        observation,
    )
    run_unit(root, "A2", 17, observation)
    return root


def _unit(root: Path) -> tuple[Path, dict]:
    path = root / "units/A2/seed-17"
    return path, json.loads((path / "unit-manifest.json").read_text())


def _reseal_unit(path: Path, manifest: dict) -> None:
    manifest["unit_manifest_sha256"] = unit_identity(manifest)
    (path / "unit-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


@pytest.mark.parametrize(
    "relative",
    ["training/initialization.pt", "training/trained.pt", "training/optimizer-state.pt"],
)
def test_tensor_artifact_mutation_rejected(sealed_unit, tmp_path, relative) -> None:
    root = tmp_path / "copy"
    shutil.copytree(sealed_unit, root)
    path, manifest = _unit(root)
    artifact = path / relative
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises((ValueError, RuntimeError)):
        validate_unit_artifacts(path, manifest)


@pytest.mark.parametrize(
    ("relative", "checkpoint_hash_field"),
    [
        ("training/training-config.json", "config_hash"),
        ("training/training-report.json", "training_report_file_sha256"),
        ("training/optimizer-evidence.json", "optimizer_evidence_file_sha256"),
    ],
)
def test_resealed_claim_artifact_mutation_rejected(
    sealed_unit, tmp_path, relative, checkpoint_hash_field
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(sealed_unit, root)
    path, manifest = _unit(root)
    artifact = path / relative
    value = json.loads(artifact.read_text())
    value["unexpected_review_mutation"] = True
    artifact.write_text(json.dumps(value))
    checkpoint_path = path / "training/checkpoint-manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint[checkpoint_hash_field] = sha256_bytes(artifact.read_bytes())
    checkpoint_path.write_text(json.dumps(checkpoint))
    manifest["checkpoint_manifest_sha256"] = sha256_bytes(checkpoint_path.read_bytes())
    _reseal_unit(path, manifest)
    with pytest.raises((ValueError, ValidationError)):
        validate_unit_artifacts(path, manifest)


def test_resealed_task_result_mutation_rejected(sealed_unit, tmp_path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(sealed_unit, root)
    path, manifest = _unit(root)
    results = path / "task-results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    rows[0]["goal_reached"] = not rows[0]["goal_reached"]
    results.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest["task_results_sha256"] = sha256_bytes(results.read_bytes())
    _reseal_unit(path, manifest)
    with pytest.raises((ValueError, ValidationError)):
        validate_unit_artifacts(path, manifest)


def test_evidence_mutation_rejected(sealed_unit, tmp_path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(sealed_unit, root)
    path, manifest = _unit(root)
    evidence = next((path / "evidence").glob("*/episode-log.json"))
    evidence.write_bytes(evidence.read_bytes() + b" ")
    with pytest.raises((ValueError, ValidationError)):
        validate_unit_artifacts(path, manifest)


def test_missing_and_extra_artifacts_rejected(sealed_unit, tmp_path) -> None:
    for mode in ("missing", "extra"):
        root = tmp_path / mode
        shutil.copytree(sealed_unit, root)
        path, manifest = _unit(root)
        if mode == "missing":
            (path / "training/training-report.json").unlink()
        else:
            (path / "extra.bin").write_bytes(b"x")
        with pytest.raises(ValueError, match="COVERAGE_MISMATCH"):
            validate_unit_artifacts(path, manifest)


def test_other_attempt_and_execution_commit_rejected(sealed_unit, tmp_path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(sealed_unit, root)
    attempt_path = root / "attempt-id.json"
    attempt = json.loads(attempt_path.read_text())
    attempt["execution_implementation_commit"] = "1" * 40
    attempt["attempt_identity_sha256"] = attempt_identity(attempt)
    attempt_path.write_text(json.dumps(attempt))
    with pytest.raises(ValueError):
        validate_attempt(root)


def test_stale_checkout_rejected(sealed_unit, monkeypatch) -> None:
    monkeypatch.setattr("scripts.fixed_target_quality_sharded.checkout_commit", lambda: "2" * 40)
    with pytest.raises(ValueError, match="NOT_CHECKED_OUT"):
        validate_attempt(sealed_unit)


def test_unit_from_another_attempt_rejected(sealed_unit) -> None:
    path, unit_manifest = _unit(sealed_unit)
    attempt = json.loads((sealed_unit / "attempt-id.json").read_text())
    contract = json.loads((sealed_unit / "target-contract.json").read_text())
    unit_manifest["attempt_identity_sha256"] = "sha256:" + "9" * 64
    unit_manifest["unit_manifest_sha256"] = unit_identity(unit_manifest)
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        validate_unit_manifest(unit_manifest, attempt, contract)


def test_fully_resealed_contradictory_unit_rejected(sealed_unit, tmp_path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(sealed_unit, root)
    path, unit_manifest = _unit(root)
    unit_manifest["seed"] = 29
    _reseal_unit(path, unit_manifest)
    with pytest.raises(ValueError):
        validate_unit_artifacts(path, unit_manifest)


def test_fully_resealed_initialization_lineage_rejected(sealed_unit, tmp_path) -> None:
    import torch

    from planner_toy.numeric_identity import canonical_state_dict_sha256
    from planner_toy.training import state_dict_sha256

    root = tmp_path / "copy"
    shutil.copytree(sealed_unit, root)
    path, unit_manifest = _unit(root)
    initialization_path = path / "training/initialization.pt"
    initialization = torch.load(initialization_path, map_location="cpu", weights_only=True)
    first = next(iter(initialization))
    initialization[first] = initialization[first].clone()
    initialization[first].view(-1)[0] += 1.0
    torch.save(initialization, initialization_path)
    checkpoint_path = path / "training/checkpoint-manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["initialization_file_sha256"] = sha256_bytes(initialization_path.read_bytes())
    checkpoint["initialization_state_dict_sha256"] = state_dict_sha256(initialization)
    checkpoint["canonical_initialization_state_dict_sha256"] = canonical_state_dict_sha256(
        initialization
    )
    checkpoint_path.write_text(json.dumps(checkpoint))
    unit_manifest["checkpoint_manifest_sha256"] = sha256_bytes(checkpoint_path.read_bytes())
    _reseal_unit(path, unit_manifest)
    with pytest.raises(ValueError, match="REPLAY_MISMATCH"):
        validate_unit_artifacts(path, unit_manifest)


def test_resealed_checkpoint_extra_claim_rejected(sealed_unit, tmp_path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(sealed_unit, root)
    path, unit_manifest = _unit(root)
    checkpoint_path = path / "training/checkpoint-manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["unreviewed_claim"] = "forged"
    checkpoint_path.write_text(json.dumps(checkpoint))
    unit_manifest["checkpoint_manifest_sha256"] = sha256_bytes(checkpoint_path.read_bytes())
    _reseal_unit(path, unit_manifest)
    with pytest.raises(ValidationError):
        validate_unit_artifacts(path, unit_manifest)


@pytest.mark.parametrize(
    "field",
    [
        "os_version",
        "runner_image",
        "cpu_model",
        "microcode",
        "kernel_version",
        "python_version",
        "torch_version",
        "torch_build_configuration_sha256",
        "actual_atten_cpu_capability",
        "torch_num_threads",
        "deterministic_algorithms",
        "mkldnn_enabled",
    ],
)
def test_resealed_forged_observation_rejected(sealed_unit, tmp_path, field) -> None:
    root = tmp_path / "copy"
    shutil.copytree(sealed_unit, root)
    observation_path = root / "initial-observation.json"
    observation = json.loads(observation_path.read_text())
    observation[field] = "forged"
    observation["observation_sha256"] = ""
    from scripts.fixed_target_contract import observation_sha256

    observation["observation_sha256"] = observation_sha256(observation)
    observation_path.write_text(json.dumps(observation))
    attempt_path = root / "attempt-id.json"
    attempt = json.loads(attempt_path.read_text())
    attempt["target_observation_sha256"] = observation["observation_sha256"]
    attempt["attempt_identity_sha256"] = attempt_identity(attempt)
    attempt_path.write_text(json.dumps(attempt))
    with pytest.raises(ValueError, match="NOT_LIVE|OBSERVATION_FORGED"):
        validate_attempt(root)


def test_runtime11_training_resets_rng_before_model(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(
        "scripts.fixed_target_quality_sharded.q.configure_canonical_cpu_runtime",
        lambda seed: calls.append(seed),
    )
    monkeypatch.setattr(
        "scripts.fixed_target_quality_sharded.q.LockedPlanner",
        lambda *_: (_ for _ in ()).throw(RuntimeError("stop-after-reset")),
    )
    with pytest.raises(RuntimeError, match="stop-after-reset"):
        _train_runtime11([], "A2", 29, tmp_path, "sha256:" + "0" * 64)
    assert calls == [29]
