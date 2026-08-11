from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

import scripts.fixed_target_contract as ft
from scripts.fixed_target_contract import (
    CLAIM_IDENTITY_FIELDS,
    EXECUTION_EVIDENCE_VERSION,
    FIXED_TARGET_ACCEPTANCE_VERSION,
    HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
    RUNTIME_1_1_EXECUTION_GATE,
    TARGET_CONTRACT_VERSION,
    TARGET_OBSERVATION_VERSION,
    acceptance_identity_sha256,
    attempt_manifest_sha256,
    build_runtime_contract,
    build_scientific_policy,
    canonical_result_identity,
    execution_evidence_sha256,
    fixed_target_source_paths,
    observation_sha256,
    require_semantic_validation_checkout,
    require_trusted_implementation_commit,
    runtime_contract_sha256,
    sharded_source_inventory_at_commit,
    source_inventory_at_commit,
    target_contract_sha256,
    validate_acceptance_bundle,
    validate_acceptance_record,
    validate_attempt_manifest,
    validate_execution_binding_contract,
    validate_execution_evidence_manifest,
    validate_observation_against_contract,
    validate_runtime_contract,
    validate_scientific_policy,
    validate_sharded_source_inventory,
    validate_target_contract,
    validate_target_observation,
)

H = "sha256:" + "1" * 64
H2 = "sha256:" + "2" * 64
H3 = "sha256:" + "3" * 64
COMMIT = "a" * 40


def valid_contract() -> dict:
    return {
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "target_name": "planning-model-canonical-cpu-v1",
        "os": "Linux",
        "os_version": "Ubuntu 24.04.4 LTS",
        "kernel_policy": {"mode": "exact", "value": "6.8.0-71-generic"},
        "architecture": "x86_64",
        "cpu_vendor": "GenuineIntel",
        "cpu_family": "6",
        "cpu_model": "85",
        "cpu_stepping": "7",
        "cpu_model_name": "Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz",
        "microcode_policy": {"mode": "exact", "value": "0x5003707"},
        "required_cpu_flags": ["avx", "avx2", "fma", "sse2"],
        "forbidden_or_irrelevant_flags_policy": {
            "mode": "ignore-unlisted-reject-forbidden",
            "forbidden_flags": [],
        },
        "logical_cpu_count_policy": {"mode": "exact", "value": 4},
        "runner_type": "self-hosted-dedicated",
        "required_runner_labels": [
            "linux",
            "planning-model-canonical-cpu-v1",
            "self-hosted",
            "x64",
        ],
        "runner_image": "planning-model-cpu-v1-image-sha256-0123456789abcdef",
        "python_implementation": "CPython",
        "python_version": "3.11.15",
        "python_build": ["main", "Aug 1 2026 00:00:00"],
        "python_compiler": "GCC 13.3.0",
        "pip_version": "26.1.2",
        "torch_version": "2.12.0+cpu",
        "torch_build_configuration_sha256": H,
        "mkl_available": True,
        "openmp_available": True,
        "mkldnn_available": True,
        "ATEN_CPU_CAPABILITY": "avx2",
        "actual_atten_cpu_capability": "AVX2",
        "MKL_CBWR": "COMPATIBLE",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "mkldnn_enabled": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "optimizer_foreach": False,
        "optimizer_fused": False,
    }


def valid_observation(contract: dict | None = None) -> dict:
    contract = contract or valid_contract()
    observation = {
        "observation_version": TARGET_OBSERVATION_VERSION,
        "target_contract_version": contract["target_contract_version"],
        "target_contract_sha256": target_contract_sha256(contract),
        "os": contract["os"],
        "os_version": contract["os_version"],
        "kernel_version": contract["kernel_policy"]["value"],
        "architecture": contract["architecture"],
        "cpu_vendor": contract["cpu_vendor"],
        "cpu_family": contract["cpu_family"],
        "cpu_model": contract["cpu_model"],
        "cpu_stepping": contract["cpu_stepping"],
        "cpu_model_name": contract["cpu_model_name"],
        "microcode": contract["microcode_policy"]["value"],
        "cpu_flags": sorted(contract["required_cpu_flags"] + ["aes", "cx8"]),
        "logical_cpu_count": contract["logical_cpu_count_policy"]["value"],
        "runner_type": contract["runner_type"],
        "runner_image": contract["runner_image"],
        "python_implementation": contract["python_implementation"],
        "python_version": contract["python_version"],
        "python_build": contract["python_build"],
        "python_compiler": contract["python_compiler"],
        "pip_version": contract["pip_version"],
        "torch_version": contract["torch_version"],
        "torch_build_configuration_sha256": contract["torch_build_configuration_sha256"],
        "mkl_available": contract["mkl_available"],
        "openmp_available": contract["openmp_available"],
        "mkldnn_available": contract["mkldnn_available"],
        "ATEN_CPU_CAPABILITY": contract["ATEN_CPU_CAPABILITY"],
        "actual_atten_cpu_capability": contract["actual_atten_cpu_capability"],
        "MKL_CBWR": contract["MKL_CBWR"],
        "torch_num_threads": contract["torch_num_threads"],
        "torch_num_interop_threads": contract["torch_num_interop_threads"],
        "OMP_NUM_THREADS": contract["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": contract["MKL_NUM_THREADS"],
        "OPENBLAS_NUM_THREADS": contract["OPENBLAS_NUM_THREADS"],
        "NUMEXPR_NUM_THREADS": contract["NUMEXPR_NUM_THREADS"],
        "mkldnn_enabled": contract["mkldnn_enabled"],
        "deterministic_algorithms": contract["deterministic_algorithms"],
        "deterministic_warn_only": contract["deterministic_warn_only"],
        "observation_sha256": "",
    }
    observation["observation_sha256"] = observation_sha256(observation)
    return observation


def claims(seed: str = "1") -> dict[str, str]:
    return {field: "sha256:" + seed * 64 for field in CLAIM_IDENTITY_FIELDS}


def valid_acceptance(*, accepted: bool = True) -> dict:
    contract = valid_contract()
    runtime = build_runtime_contract(contract)
    target_hash = target_contract_sha256(contract)
    runtime_hash = runtime_contract_sha256(runtime)
    common_claims = claims()
    attempts = []
    for index in range(1, 4):
        observation = valid_observation(contract)
        attempts.append(
            {
                "attempt_index": index,
                "workflow_run_id": 1000 + index,
                "job_id": 2000 + index,
                "implementation_commit": COMMIT,
                "target_contract_sha256": target_hash,
                "runtime_contract_sha256": runtime_hash,
                "target_observation": observation,
                "target_observation_sha256": observation["observation_sha256"],
                "source_inventory_sha256": H2,
                "observed_optimizer_foreach": False,
                "observed_optimizer_fused": False,
                "execution_evidence_sha256": H2,
                "canonical_result_identity": canonical_result_identity(common_claims),
                "claim_identities": copy.deepcopy(common_claims),
                "training_execution_mode": "TRAINED_IN_RUN",
                "checkpoint_origin_run_hash": None,
                "reuse_source_manifest_hash": None,
                "successful_full_evaluation": True,
                "result": "PASS",
            }
        )
    value = {
        "acceptance_version": FIXED_TARGET_ACCEPTANCE_VERSION,
        "status": "FIXED_TARGET_ACCEPTED" if accepted else "TARGET_PROVISIONED",
        "implementation_commit": COMMIT,
        "target_contract": contract,
        "target_contract_sha256": target_hash,
        "runtime_contract": runtime,
        "runtime_contract_sha256": runtime_hash,
        "attempts": attempts,
        "cross_attempt_comparison": {
            "status": "PASS" if accepted else "NOT_RUN",
            "mismatches": [],
        },
        "accepted": accepted,
        "blocker": None,
        "acceptance_identity": "",
    }
    value["acceptance_identity"] = acceptance_identity_sha256(value)
    return value


def blocked_acceptance() -> dict:
    value = {
        "acceptance_version": FIXED_TARGET_ACCEPTANCE_VERSION,
        "status": "BLOCKED_ON_FIXED_RUNNER_PROVISIONING",
        "implementation_commit": None,
        "target_contract": None,
        "target_contract_sha256": None,
        "runtime_contract": None,
        "runtime_contract_sha256": None,
        "attempts": [],
        "cross_attempt_comparison": {"status": "NOT_RUN", "mismatches": []},
        "accepted": False,
        "blocker": {
            "code": "FIXED_RUNNER_NOT_PROVISIONED",
            "required_action": "Provision dedicated self-hosted Linux x86_64 runner.",
        },
        "acceptance_identity": "",
    }
    value["acceptance_identity"] = acceptance_identity_sha256(value)
    return value


def reseal(value: dict) -> None:
    value["acceptance_identity"] = acceptance_identity_sha256(value)


def test_contract_runtime_and_preflight_observation_are_semantically_separate() -> None:
    contract = valid_contract()
    validate_target_contract(contract)
    runtime = build_runtime_contract(contract)
    validate_runtime_contract(runtime, contract)
    observation = valid_observation(contract)
    validate_target_observation(observation)
    validate_observation_against_contract(contract, observation)
    assert "optimizer_foreach" not in observation
    assert "optimizer_fused" not in observation
    assert "runner_labels" not in observation
    assert contract["required_runner_labels"]


def test_preflight_cannot_self_attest_optimizer_execution() -> None:
    observation = valid_observation()
    observation["optimizer_foreach"] = False
    observation["optimizer_fused"] = False
    observation["observation_sha256"] = observation_sha256(observation)
    with pytest.raises(ValueError, match="FIXED_TARGET_OBSERVATION_SCHEMA_INVALID"):
        validate_target_observation(observation)


def test_fully_synthetic_three_of_three_record_requires_bundle() -> None:
    with pytest.raises(ValueError, match=RUNTIME_1_1_EXECUTION_GATE):
        validate_acceptance_record(valid_acceptance())


def test_provisional_record_still_checks_duplicate_run_ids() -> None:
    value = valid_acceptance(accepted=False)
    value["attempts"][1]["workflow_run_id"] = value["attempts"][0]["workflow_run_id"]
    reseal(value)
    with pytest.raises(ValueError, match="FIXED_TARGET_ACCEPTANCE_DUPLICATE_WORKFLOW_RUN"):
        validate_acceptance_record(value)


@pytest.mark.parametrize("field", ["training_execution_mode", "source_inventory_sha256"])
def test_three_of_three_rejects_cross_attempt_provenance_drift(field: str) -> None:
    value = valid_acceptance()
    value["attempts"][1][field] = (
        "TRAINED_IN_ATTEMPT_SHARDED" if field == "training_execution_mode" else H3
    )
    reseal(value)
    with pytest.raises(ValueError, match="CROSS_ATTEMPT_COMPARISON_MISMATCH"):
        validate_acceptance_record(value)


def test_three_of_three_rejects_cross_attempt_observation_drift() -> None:
    value = valid_acceptance()
    observation = copy.deepcopy(value["attempts"][1]["target_observation"])
    observation["cpu_flags"].append("permitted-extra-flag")
    observation["cpu_flags"].sort()
    observation["observation_sha256"] = observation_sha256(observation)
    value["attempts"][1]["target_observation"] = observation
    value["attempts"][1]["target_observation_sha256"] = observation["observation_sha256"]
    reseal(value)
    with pytest.raises(ValueError, match="CROSS_ATTEMPT_COMPARISON_MISMATCH"):
        validate_acceptance_record(value)


def test_blocked_record_remains_valid() -> None:
    validate_acceptance_record(blocked_acceptance())


def test_blocked_record_cannot_contain_attempts_or_contracts() -> None:
    value = blocked_acceptance()
    value["target_contract"] = valid_contract()
    reseal(value)
    with pytest.raises(ValueError, match="FIXED_TARGET_ACCEPTANCE_SCHEMA_INVALID"):
        validate_acceptance_record(value)


def _write_bundle_shell(root: Path, acceptance: dict) -> None:
    (root / "acceptance.json").write_text(json.dumps(acceptance), encoding="utf-8")
    for index in range(1, 4):
        attempt = root / f"attempt-{index}"
        attempt.mkdir()
        (attempt / "evaluation").mkdir()
        (attempt / "attempt_manifest.json").write_text("{}", encoding="utf-8")
        (attempt / "preflight.json").write_text("{}", encoding="utf-8")
        (attempt / "execution-evidence.json").write_text("{}", encoding="utf-8")


def test_final_bundle_is_explicitly_gated_before_runtime_1_1_semantics(
    tmp_path: Path,
) -> None:
    acceptance = valid_acceptance()
    _write_bundle_shell(tmp_path, acceptance)
    with pytest.raises(ValueError, match=RUNTIME_1_1_EXECUTION_GATE):
        validate_acceptance_bundle(tmp_path)


def _attempt_shape(root: Path) -> Path:
    attempt = root / "attempt-1"
    evaluation = attempt / "evaluation"
    (evaluation / "training-runs").mkdir(parents=True)
    (evaluation / "evidence").mkdir()
    for name in ft._TOP_LEVEL_EVALUATION_FILES:
        (evaluation / name).write_bytes(b"x")
    for variant in ("A2", "A3", "A4"):
        for seed in (17, 29, 43):
            run_dir = evaluation / "training-runs" / variant / f"seed-{seed}"
            run_dir.mkdir(parents=True)
            for name in ft._TRAINING_RUN_FILES:
                (run_dir / name).write_bytes(b"x")
    (attempt / "preflight.json").write_bytes(b"{}")
    (attempt / "execution-evidence.json").write_bytes(b"{}")
    manifest = {
        "attempt_manifest_version": ft.ATTEMPT_MANIFEST_VERSION,
        "attempt_index": 1,
        "files": {},
        "attempt_manifest_sha256": "",
    }
    manifest["files"] = ft._relative_file_hashes(attempt, exclude={"attempt_manifest.json"})
    manifest["attempt_manifest_sha256"] = attempt_manifest_sha256(manifest)
    (attempt / "attempt_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return attempt


def test_missing_attempt_file_rejected(tmp_path: Path) -> None:
    attempt = _attempt_shape(tmp_path)
    (attempt / "evaluation/training-runs/A2/seed-17/trained.pt").unlink()
    with pytest.raises(ValueError, match="FIXED_TARGET_TRAINING_RUN_FILE_COVERAGE_MISMATCH"):
        ft._validate_attempt_shape(attempt)


def test_extra_claim_bearing_attempt_file_rejected(tmp_path: Path) -> None:
    attempt = _attempt_shape(tmp_path)
    (attempt / "evaluation/training-runs/A2/seed-17/extra-checkpoint.pt").write_bytes(b"x")
    with pytest.raises(ValueError, match="FIXED_TARGET_TRAINING_RUN_FILE_COVERAGE_MISMATCH"):
        ft._validate_attempt_shape(attempt)


@pytest.mark.parametrize(
    "relative",
    [
        "evaluation/training-runs/A2/seed-17/trained.pt",
        "evaluation/training-runs/A2/seed-17/optimizer-state.pt",
    ],
)
def test_modified_claim_file_rejected_even_if_outer_acceptance_resealed(
    tmp_path: Path,
    relative: str,
) -> None:
    attempt = _attempt_shape(tmp_path)
    manifest = json.loads((attempt / "attempt_manifest.json").read_text())
    (attempt / relative).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="FIXED_TARGET_ATTEMPT_MANIFEST_COVERAGE_MISMATCH"):
        validate_attempt_manifest(attempt, manifest, 1)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _temp_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    (repo / "planner_toy").mkdir()
    (repo / "planner_toy/__init__.py").write_text('"""x"""\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_nonexistent_forty_hex_implementation_sha_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _temp_repo(tmp_path)
    monkeypatch.setattr(ft, "ROOT", repo)
    with pytest.raises(ValueError, match="FIXED_TARGET_SOURCE_COMMIT_NOT_FOUND"):
        source_inventory_at_commit("a" * 40)


def test_source_inventory_mutation_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, first = _temp_repo(tmp_path)
    monkeypatch.setattr(ft, "ROOT", repo)
    monkeypatch.setattr(
        ft,
        "fixed_target_source_paths",
        lambda: ("planner_toy/__init__.py", "pyproject.toml"),
    )
    old = source_inventory_at_commit(first)
    (repo / "pyproject.toml").write_text("[build-system]\nrequires=[]\n", encoding="utf-8")
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-m", "mutate pyproject")
    second = _git(repo, "rev-parse", "HEAD")
    new = source_inventory_at_commit(second)
    assert old["source_inventory_sha256"] != new["source_inventory_sha256"]
    with pytest.raises(ValueError, match="FIXED_TARGET_SOURCE_INVENTORY_MISMATCH"):
        ft.validate_source_inventory(old, implementation_commit=second)


def test_sharded_inventory_is_closed_sorted_and_rejects_legacy_downgrade() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    inventory = sharded_source_inventory_at_commit(commit)
    paths = [entry["path"] for entry in inventory["files"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    validate_sharded_source_inventory(inventory, implementation_commit=commit)
    with pytest.raises(ValueError, match="SHARDED_SOURCE_INVENTORY_MISMATCH"):
        validate_sharded_source_inventory(
            source_inventory_at_commit(commit), implementation_commit=commit
        )
    reordered = copy.deepcopy(inventory)
    reordered["files"] = list(reversed(reordered["files"]))
    reordered["source_inventory_sha256"] = ft.sha256_value(reordered["files"])
    with pytest.raises(ValueError, match="SHARDED_SOURCE_INVENTORY_MISMATCH"):
        validate_sharded_source_inventory(reordered, implementation_commit=commit)


def test_pyproject_and_package_initializer_change_source_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, first = _temp_repo(tmp_path)
    monkeypatch.setattr(ft, "ROOT", repo)
    monkeypatch.setattr(
        ft,
        "fixed_target_source_paths",
        lambda: ("planner_toy/__init__.py", "pyproject.toml"),
    )
    first_hash = source_inventory_at_commit(first)["source_inventory_sha256"]
    (repo / "planner_toy/__init__.py").write_text('"""changed"""\n', encoding="utf-8")
    _git(repo, "add", "planner_toy/__init__.py")
    _git(repo, "commit", "-m", "mutate initializer")
    second = _git(repo, "rev-parse", "HEAD")
    assert source_inventory_at_commit(second)["source_inventory_sha256"] != first_hash


def test_fixed_target_inventory_is_exact_quality_lock_plus_transitive_additions() -> None:
    locked = json.loads(ft.QUALITY_LOCK_PATH.read_text(encoding="utf-8"))
    quality_paths = {entry["path"] for entry in locked["evaluator_source_files"]}
    expected = quality_paths | ft._FIXED_TARGET_SOURCE_ADDITIONS
    assert set(fixed_target_source_paths()) == expected
    assert "requirements.lock" in expected
    assert "pyproject.toml" in expected
    assert "planner_toy/__init__.py" in expected
    assert "scripts/__init__.py" in expected
    assert ft.QUALITY_LOCK_RELATIVE_PATH in expected


def _write_optimizer_state(root: Path, foreach: object, fused: object) -> None:
    import torch

    run_dir = root / "training-runs/A2/seed-17"
    run_dir.mkdir(parents=True)
    torch.save(
        {
            "state": {},
            "param_groups": [{"params": [], "foreach": foreach, "fused": fused}],
        },
        run_dir / "optimizer-state.pt",
    )


def test_actual_optimizer_none_none_rejected_against_required_false_false(
    tmp_path: Path,
) -> None:
    _write_optimizer_state(tmp_path, None, None)
    with pytest.raises(ValueError, match="FIXED_TARGET_OPTIMIZER_EXECUTION_MISMATCH"):
        ft._derive_optimizer_execution(tmp_path)


def test_actual_optimizer_false_false_accepted(tmp_path: Path) -> None:
    _write_optimizer_state(tmp_path, False, False)
    assert ft._derive_optimizer_execution(tmp_path) == (False, False)


def test_untrusted_implementation_not_reachable_from_protected_main_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, main_commit = _temp_repo(tmp_path)
    _git(repo, "update-ref", "refs/remotes/origin/main", main_commit)
    _git(repo, "checkout", "--orphan", "side")
    for child in list(repo.iterdir()):
        if child.name != ".git":
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    (repo / "side.txt").write_text("side", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "side")
    side = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(ft, "ROOT", repo)
    with pytest.raises(ValueError, match="FIXED_TARGET_IMPLEMENTATION_NOT_TRUSTED"):
        require_trusted_implementation_commit(side)
    assert require_trusted_implementation_commit(main_commit) == main_commit


def test_frozen_v0_1_artifact_blob_guards_when_running_in_repo() -> None:
    root = Path(__file__).resolve().parents[2]
    guards = {
        "docs/evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md": "e2344c07a76fcf7de140f894317fb509f6bc04fb",  # noqa: E501
        "docs/evaluations/data/a2_a3_a4_heldout_summary.json": "408742e15a3cddacdefcb0f0b814a6d68a5ca62d",  # noqa: E501
        "docs/evaluations/A2_A3_A4_V0_1_DECISION_RU.md": "909bf35b65b1e7b1e00f2366519b776333b473b2",
        ".github/workflows/ci.yml": "36463b4c005e9deb71adbd9ba9faea6603ebdaf2",
        "planner_toy/canonical_runtime.py": "057cfbf29ed486659a6ba7b036cdd740d1bb9b44",
        "planner_toy/quality.py": "d1b4b48a94a75176f31f074a17c7bb3bcbf644de",
    }
    if not (root / ".git").exists():
        pytest.skip("full git checkout required for frozen blob guard")
    for relative, expected in guards.items():
        completed = subprocess.run(
            ["git", "hash-object", relative],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        assert completed.stdout.strip() == expected


def _valid_execution_evidence(acceptance: dict, attempt: dict) -> dict:
    policy = build_scientific_policy()
    value = {
        "execution_evidence_version": EXECUTION_EVIDENCE_VERSION,
        "implementation_commit": attempt["implementation_commit"],
        "target_contract_sha256": attempt["target_contract_sha256"],
        "runtime_contract_sha256": attempt["runtime_contract_sha256"],
        "target_observation_sha256": attempt["target_observation_sha256"],
        "source_inventory_sha256": attempt["source_inventory_sha256"],
        "scientific_policy": policy,
        "scientific_policy_sha256": policy["scientific_policy_sha256"],
        "evaluator_version": "future-runtime-1.1-evaluator/1.0",
        "evaluator_source_sha256": H3,
        "requirements_lock_sha256": H,
        "dataset_identity": {
            "dataset_hash": policy["dataset_hash"],
            "ordered_train_task_ids": list(policy["ordered_train_task_ids"]),
            "ordered_eval_task_ids": list(policy["ordered_eval_task_ids"]),
        },
        "variants": ["A2", "A3", "A4"],
        "seeds": [17, 29, 43],
        "epochs": 3,
        "updates_per_run": 9,
        "optimizer_class": "AdamW",
        "optimizer_hyperparameters": {
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
        "gradient_clipping": {"algorithm": "clip_grad_norm_", "max_norm": 1.0},
        "observed_optimizer_foreach": False,
        "observed_optimizer_fused": False,
        "evaluation_root_identity": H2,
        "execution_evidence_sha256": "",
    }
    value["execution_evidence_sha256"] = execution_evidence_sha256(value)
    validate_execution_evidence_manifest(value)
    return value


@pytest.mark.parametrize(
    "missing",
    ["scientific_parent_implementation_commit", "execution_topology", "execution_context"],
)
def test_sharded_execution_evidence_requires_sharded_provenance(missing: str) -> None:
    acceptance = valid_acceptance(accepted=False)
    evidence = _valid_execution_evidence(acceptance, acceptance["attempts"][0])
    evidence.update(
        {
            "evaluator_version": "development-quality-evaluation/0.1-runtime1.1-sharded/1.0",
            "scientific_parent_implementation_commit": (
                HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT
            ),
            "execution_topology": "SHARDED_VARIANT_SEED_SUBPROCESSES",
            "execution_context": "formal-fixed-target",
        }
    )
    evidence.pop(missing)
    evidence["execution_evidence_sha256"] = execution_evidence_sha256(evidence)
    with pytest.raises(ValueError, match="EXECUTION_EVIDENCE_SCHEMA_INVALID"):
        validate_execution_evidence_manifest(evidence)


def _binding_inputs(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict, dict, list[dict]]:
    acceptance = valid_acceptance(accepted=False)
    attempt = acceptance["attempts"][0]
    evidence = _valid_execution_evidence(acceptance, attempt)
    attempt["execution_evidence_sha256"] = evidence["execution_evidence_sha256"]
    reseal(acceptance)
    preflight = {
        "implementation_commit": attempt["implementation_commit"],
        "target_contract_sha256": attempt["target_contract_sha256"],
        "runtime_contract_sha256": attempt["runtime_contract_sha256"],
        "target_observation": attempt["target_observation"],
        "source_inventory": {"source_inventory_sha256": attempt["source_inventory_sha256"]},
    }
    policy = evidence["scientific_policy"]
    optimizer = {
        "name": "AdamW",
        "learning_rate": 3e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
    }
    evaluation_config = {
        "implementation_commit": attempt["implementation_commit"],
        "evaluator_version": evidence["evaluator_version"],
        "evaluator_source_sha256": evidence["evaluator_source_sha256"],
        "requirements_lock_sha256": H,
        "dataset_manifest_hash": policy["dataset_hash"],
        "train_task_ids": evidence["dataset_identity"]["ordered_train_task_ids"],
        "eval_task_ids": evidence["dataset_identity"]["ordered_eval_task_ids"],
        "variants": ["A2", "A3", "A4"],
        "seeds": [17, 29, 43],
        "optimizer": optimizer,
        "checkpoint_policy": policy["checkpoint_policy"],
        "training_execution_mode": "TRAINED_IN_RUN",
    }
    training_configs = []
    for variant in ("A2", "A3", "A4"):
        for seed in (17, 29, 43):
            training_configs.append(
                {
                    "variant_identity": {"implementation_variant": variant},
                    "seed": seed,
                    "dataset_hash": policy["dataset_hash"],
                    "train_task_ids": evidence["dataset_identity"]["ordered_train_task_ids"],
                    "epochs": 3,
                    "updates": 9,
                    "optimizer": optimizer,
                    "checkpoint_policy": policy["checkpoint_policy"],
                }
            )
    monkeypatch.setattr(ft, "requirements_lock_sha256_at_commit", lambda commit: H)
    return acceptance, evidence, preflight, evaluation_config, training_configs


def test_sharded_runtime11_execution_binding_is_explicit_and_closed(monkeypatch) -> None:
    acceptance, evidence, preflight, config, training = _binding_inputs(monkeypatch)
    attempt = acceptance["attempts"][0]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    inventory = sharded_source_inventory_at_commit(commit)
    acceptance["implementation_commit"] = commit
    attempt["implementation_commit"] = commit
    preflight["implementation_commit"] = commit
    preflight["source_inventory"] = inventory
    attempt["source_inventory_sha256"] = inventory["source_inventory_sha256"]
    evidence["implementation_commit"] = commit
    evidence["source_inventory_sha256"] = inventory["source_inventory_sha256"]
    config["implementation_commit"] = commit
    evidence.update(
        {
            "execution_topology": "SHARDED_VARIANT_SEED_SUBPROCESSES",
            "execution_context": "formal-fixed-target",
            "evaluator_source_sha256": evidence["source_inventory_sha256"],
            "scientific_parent_implementation_commit": (HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT),
            "evaluator_version": ("development-quality-evaluation/0.1-runtime1.1-sharded/1.0"),
        }
    )
    evidence["execution_evidence_sha256"] = execution_evidence_sha256(evidence)
    config.update(
        {
            "evaluator_version": evidence["evaluator_version"],
            "evaluator_source_sha256": evidence["evaluator_source_sha256"],
            "training_execution_mode": "TRAINED_IN_ATTEMPT_SHARDED",
            "execution_topology": "SHARDED_VARIANT_SEED_SUBPROCESSES",
            "execution_context": "formal-fixed-target",
            "scientific_parent_implementation_commit": (HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT),
            "target_contract_sha256": evidence["target_contract_sha256"],
            "runtime_contract_sha256": evidence["runtime_contract_sha256"],
            "target_observation_sha256": evidence["target_observation_sha256"],
            "source_inventory_sha256": evidence["source_inventory_sha256"],
            "scientific_policy_sha256": evidence["scientific_policy_sha256"],
        }
    )
    attempt["training_execution_mode"] = "TRAINED_IN_ATTEMPT_SHARDED"
    validate_execution_binding_contract(
        evidence,
        acceptance=acceptance,
        attempt=attempt,
        preflight=preflight,
        evaluation_config=config,
        target_observation=attempt["target_observation"],
        training_configs=training,
    )
    qualification_evidence = copy.deepcopy(evidence)
    qualification_config = copy.deepcopy(config)
    qualification_evidence["execution_context"] = "qualification-only"
    qualification_config["execution_context"] = "qualification-only"
    qualification_evidence["execution_evidence_sha256"] = execution_evidence_sha256(
        qualification_evidence
    )
    validate_execution_evidence_manifest(qualification_evidence)
    with pytest.raises(ValueError, match="SHARDED_EXECUTION_BINDING_MISMATCH"):
        validate_execution_binding_contract(
            qualification_evidence,
            acceptance=acceptance,
            attempt=attempt,
            preflight=preflight,
            evaluation_config=qualification_config,
            target_observation=attempt["target_observation"],
            training_configs=training,
        )
    for field, value in (
        ("execution_topology", "MONOLITHIC"),
        ("evaluator_version", "development-quality-evaluation/0.1"),
    ):
        broken = copy.deepcopy(evidence)
        broken[field] = value
        broken["execution_evidence_sha256"] = execution_evidence_sha256(broken)
        with pytest.raises(ValueError):
            validate_execution_binding_contract(
                broken,
                acceptance=acceptance,
                attempt=attempt,
                preflight=preflight,
                evaluation_config=config,
                target_observation=attempt["target_observation"],
                training_configs=training,
            )
    downgraded_evidence = copy.deepcopy(evidence)
    downgraded_config = copy.deepcopy(config)
    downgraded_attempt = copy.deepcopy(attempt)
    downgraded_config["training_execution_mode"] = "TRAINED_IN_RUN"
    downgraded_attempt["training_execution_mode"] = "TRAINED_IN_RUN"
    with pytest.raises(ValueError, match="SHARDED_EXECUTION_MODE_DOWNGRADE"):
        validate_execution_binding_contract(
            downgraded_evidence,
            acceptance=acceptance,
            attempt=downgraded_attempt,
            preflight=preflight,
            evaluation_config=downgraded_config,
            target_observation=downgraded_attempt["target_observation"],
            training_configs=training,
        )


def test_legacy_execution_identity_allows_new_execution_commit() -> None:
    ft._validate_training_execution_identity(
        {
            "training_execution_mode": "TRAINED_IN_RUN",
            "evaluator_version": "development-quality-evaluation/0.1",
            "implementation_commit": "f" * 40,
        }
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {
            "evaluator_version": (
                "development-quality-evaluation/0.1-runtime1.1-sharded/1.0"
            )
        },
        {"execution_topology": "SHARDED_VARIANT_SEED_SUBPROCESSES"},
    ],
)
def test_legacy_execution_identity_rejects_sharded_signals(mutation: dict) -> None:
    config = {
        "training_execution_mode": "TRAINED_IN_RUN",
        "evaluator_version": "development-quality-evaluation/0.1",
        "implementation_commit": "f" * 40,
    }
    config.update(mutation)
    with pytest.raises(ValueError, match="SHARDED_EXECUTION_MODE_DOWNGRADE"):
        ft._validate_training_execution_identity(config)


def test_pr19_source_inventory_profile_remains_constructible() -> None:
    inventory = ft.source_inventory_at_commit("1833c2ae21618a559a34f84aae1dba292030edd5")
    assert inventory["source_inventory_version"] == ft.SOURCE_INVENTORY_VERSION
    assert not any(
        entry["path"] == "scripts/fixed_target_quality_sharded.py" for entry in inventory["files"]
    )


def test_provisional_acceptance_admits_truthful_sharded_training_mode() -> None:
    acceptance = valid_acceptance(accepted=False)
    for attempt in acceptance["attempts"]:
        attempt["training_execution_mode"] = "TRAINED_IN_ATTEMPT_SHARDED"
    reseal(acceptance)
    validate_acceptance_record(acceptance)


def test_fixed_target_schemas_do_not_enter_historical_quality_source_files() -> None:
    from planner_toy import quality

    schema_names = {path.name for path in ft.SCHEMA_ROOT.glob("fixed_target_*.schema.json")}
    assert schema_names
    assert all(not name.startswith("toy_") for name in schema_names)
    assert not any(
        path.startswith("planner_toy/schemas/fixed_target_") for path in quality.SOURCE_FILES
    )
    assert ft.QUALITY_LOCK_RELATIVE_PATH not in quality.SOURCE_FILES
    current = quality.source_identity()
    historical = quality.source_identity_at_commit(HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT)
    assert current["evaluator_source_sha256"] == ft._QUALITY_SOURCE_LOCK_SHA256
    assert historical["evaluator_source_sha256"] == ft._QUALITY_SOURCE_LOCK_SHA256


def test_historical_quality_source_lock_is_exact() -> None:
    assert ft._QUALITY_SOURCE_LOCK_SHA256 == (
        "sha256:9205ad312fc37fa9927505e9c44a599e29fc5e31180db9d2e49ebfcc247b4570"
    )
    locked = json.loads(ft.QUALITY_LOCK_PATH.read_text(encoding="utf-8"))
    assert locked["implementation_commit"] == HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT
    assert locked["evaluator_version"] == ft.HISTORICAL_QUALITY_EVALUATOR_VERSION
    assert locked["evaluator_source_sha256"] == ft._QUALITY_SOURCE_LOCK_SHA256
    assert locked["dataset_hash"] == ft.HISTORICAL_QUALITY_DATASET_HASH
    assert ft.sha256_value(locked["evaluator_source_files"]) == ft._QUALITY_SOURCE_LOCK_SHA256
    assert ft.validate_historical_quality_lock(locked) == locked


def test_fixed_target_schema_mutation_changes_fixed_target_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, first = _temp_repo(tmp_path)
    schema_dir = repo / "planner_toy/schemas"
    schema_dir.mkdir()
    schema = schema_dir / "fixed_target_runtime.schema.json"
    schema.write_text('{"v":1}\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add fixed target schema")
    first = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(ft, "ROOT", repo)
    monkeypatch.setattr(
        ft,
        "fixed_target_source_paths",
        lambda: ("planner_toy/schemas/fixed_target_runtime.schema.json", "pyproject.toml"),
    )
    before = source_inventory_at_commit(first)["source_inventory_sha256"]
    schema.write_text('{"v":2}\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "mutate fixed target schema")
    second = _git(repo, "rev-parse", "HEAD")
    after = source_inventory_at_commit(second)["source_inventory_sha256"]
    assert before != after


def test_wrong_scientific_policy_rejected() -> None:
    policy = build_scientific_policy()
    policy["epochs"] = 4
    policy["scientific_policy_sha256"] = ft.scientific_policy_sha256(policy)
    with pytest.raises(ValueError, match="FIXED_TARGET_SCIENTIFIC_POLICY"):
        validate_scientific_policy(policy)


def test_execution_implementation_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    acceptance, evidence, preflight, config, training = _binding_inputs(monkeypatch)
    evidence["implementation_commit"] = "b" * 40
    evidence["execution_evidence_sha256"] = execution_evidence_sha256(evidence)
    with pytest.raises(ValueError, match="FIXED_TARGET_EXECUTION_IMPLEMENTATION_MISMATCH"):
        validate_execution_binding_contract(
            evidence,
            acceptance=acceptance,
            attempt=acceptance["attempts"][0],
            preflight=preflight,
            evaluation_config=config,
            target_observation=acceptance["attempts"][0]["target_observation"],
            training_configs=training,
        )


def test_evaluation_execution_implementation_mismatch_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance, evidence, preflight, config, training = _binding_inputs(monkeypatch)
    config["implementation_commit"] = "b" * 40
    with pytest.raises(ValueError, match="FIXED_TARGET_EXECUTION_IMPLEMENTATION_MISMATCH"):
        validate_execution_binding_contract(
            evidence,
            acceptance=acceptance,
            attempt=acceptance["attempts"][0],
            preflight=preflight,
            evaluation_config=config,
            target_observation=acceptance["attempts"][0]["target_observation"],
            training_configs=training,
        )


def test_execution_runtime_contract_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    acceptance, evidence, preflight, config, training = _binding_inputs(monkeypatch)
    evidence["runtime_contract_sha256"] = H3
    evidence["execution_evidence_sha256"] = execution_evidence_sha256(evidence)
    with pytest.raises(ValueError, match="FIXED_TARGET_EXECUTION_RUNTIME_MISMATCH"):
        validate_execution_binding_contract(
            evidence,
            acceptance=acceptance,
            attempt=acceptance["attempts"][0],
            preflight=preflight,
            evaluation_config=config,
            target_observation=acceptance["attempts"][0]["target_observation"],
            training_configs=training,
        )


def test_execution_target_observation_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    acceptance, evidence, preflight, config, training = _binding_inputs(monkeypatch)
    evidence["target_observation_sha256"] = H3
    evidence["execution_evidence_sha256"] = execution_evidence_sha256(evidence)
    with pytest.raises(ValueError, match="FIXED_TARGET_EXECUTION_OBSERVATION_MISMATCH"):
        validate_execution_binding_contract(
            evidence,
            acceptance=acceptance,
            attempt=acceptance["attempts"][0],
            preflight=preflight,
            evaluation_config=config,
            target_observation=acceptance["attempts"][0]["target_observation"],
            training_configs=training,
        )


def test_execution_source_inventory_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    acceptance, evidence, preflight, config, training = _binding_inputs(monkeypatch)
    evidence["source_inventory_sha256"] = H3
    evidence["execution_evidence_sha256"] = execution_evidence_sha256(evidence)
    with pytest.raises(ValueError, match="FIXED_TARGET_EXECUTION_SOURCE_INVENTORY_MISMATCH"):
        validate_execution_binding_contract(
            evidence,
            acceptance=acceptance,
            attempt=acceptance["attempts"][0],
            preflight=preflight,
            evaluation_config=config,
            target_observation=acceptance["attempts"][0]["target_observation"],
            training_configs=training,
        )


def test_historical_probe_2_not_in_runtime_1_1_acceptance_contract() -> None:
    value = valid_acceptance(accepted=False)
    value["attempts"][0]["probe_identity"] = H2
    reseal(value)
    with pytest.raises(ValueError, match="FIXED_TARGET_ACCEPTANCE_SCHEMA_INVALID"):
        validate_acceptance_record(value)


def test_semantic_validation_wrong_head_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, first = _temp_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[build-system]\nrequires=[]\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "second")
    monkeypatch.setattr(ft, "ROOT", repo)
    with pytest.raises(ValueError, match="FIXED_TARGET_SEMANTIC_VALIDATION_HEAD_MISMATCH"):
        require_semantic_validation_checkout(first)


def test_semantic_validation_dirty_tracked_tree_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, commit = _temp_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[build-system]\nrequires=[]\n", encoding="utf-8")
    monkeypatch.setattr(ft, "ROOT", repo)
    with pytest.raises(ValueError, match="FIXED_TARGET_SEMANTIC_VALIDATION_DIRTY_TREE"):
        require_semantic_validation_checkout(commit)


def test_semantic_validation_exact_clean_checkout_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, commit = _temp_repo(tmp_path)
    monkeypatch.setattr(ft, "ROOT", repo)
    assert require_semantic_validation_checkout(commit) == commit


def _call_binding(
    acceptance: dict,
    evidence: dict,
    preflight: dict,
    config: dict,
    training: list[dict],
) -> None:
    validate_execution_binding_contract(
        evidence,
        acceptance=acceptance,
        attempt=acceptance["attempts"][0],
        preflight=preflight,
        evaluation_config=config,
        target_observation=acceptance["attempts"][0]["target_observation"],
        training_configs=training,
    )


def _mutate_split_and_reseal(
    evidence: dict,
    config: dict,
    training: list[dict],
    *,
    train_ids: list[str] | None = None,
    eval_ids: list[str] | None = None,
) -> None:
    if train_ids is not None:
        evidence["dataset_identity"]["ordered_train_task_ids"] = list(train_ids)
        config["train_task_ids"] = list(train_ids)
        for training_config in training:
            training_config["train_task_ids"] = list(train_ids)
    if eval_ids is not None:
        evidence["dataset_identity"]["ordered_eval_task_ids"] = list(eval_ids)
        config["eval_task_ids"] = list(eval_ids)
    evidence["execution_evidence_sha256"] = execution_evidence_sha256(evidence)


def test_real_frozen_ordered_split_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    acceptance, evidence, preflight, config, training = _binding_inputs(monkeypatch)
    assert evidence["dataset_identity"]["ordered_train_task_ids"] == list(
        ft.HISTORICAL_ORDERED_TRAIN_TASK_IDS
    )
    assert evidence["dataset_identity"]["ordered_eval_task_ids"] == list(
        ft.HISTORICAL_ORDERED_EVAL_TASK_IDS
    )
    _call_binding(acceptance, evidence, preflight, config, training)


@pytest.mark.parametrize(
    ("train_ids", "code"),
    [
        (
            ["bw-00000002", "bw-00000001", "bw-00000003"],
            "FIXED_TARGET_EXECUTION_TRAIN_SPLIT_MISMATCH",
        ),
        (["bw-00000001", "bw-00000002"], "FIXED_TARGET_EXECUTION_TRAIN_SPLIT_MISMATCH"),
    ],
)
def test_fully_resealed_wrong_train_split_rejected(
    monkeypatch: pytest.MonkeyPatch,
    train_ids: list[str],
    code: str,
) -> None:
    acceptance, evidence, preflight, config, training = _binding_inputs(monkeypatch)
    _mutate_split_and_reseal(evidence, config, training, train_ids=train_ids)
    with pytest.raises(ValueError, match=code):
        _call_binding(acceptance, evidence, preflight, config, training)


@pytest.mark.parametrize(
    "eval_ids",
    [
        ["bw-00000004", "bw-00000099"],
        ["bw-00000005", "bw-00000004"],
    ],
)
def test_fully_resealed_wrong_heldout_split_rejected(
    monkeypatch: pytest.MonkeyPatch,
    eval_ids: list[str],
) -> None:
    acceptance, evidence, preflight, config, training = _binding_inputs(monkeypatch)
    _mutate_split_and_reseal(evidence, config, training, eval_ids=eval_ids)
    with pytest.raises(ValueError, match="FIXED_TARGET_EXECUTION_EVAL_SPLIT_MISMATCH"):
        _call_binding(acceptance, evidence, preflight, config, training)


def _mutated_quality_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> dict:
    locked = json.loads(ft.QUALITY_LOCK_PATH.read_text(encoding="utf-8"))
    mutate(locked)
    path = tmp_path / "quality-lock.json"
    path.write_text(json.dumps(locked), encoding="utf-8")
    monkeypatch.setattr(ft, "QUALITY_LOCK_PATH", path)
    return locked


def test_historical_quality_lock_implementation_mutation_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mutated_quality_lock(
        tmp_path,
        monkeypatch,
        lambda locked: locked.__setitem__("implementation_commit", "b" * 40),
    )
    with pytest.raises(ValueError, match="QUALITY_SOURCE_LOCK_IMPLEMENTATION_MISMATCH"):
        build_scientific_policy()


def test_historical_quality_lock_dataset_mutation_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mutated_quality_lock(
        tmp_path,
        monkeypatch,
        lambda locked: locked.__setitem__("dataset_hash", H3),
    )
    with pytest.raises(ValueError, match="QUALITY_SOURCE_LOCK_DATASET_MISMATCH"):
        build_scientific_policy()


def test_historical_quality_lock_evaluator_version_mutation_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mutated_quality_lock(
        tmp_path,
        monkeypatch,
        lambda locked: locked.__setitem__(
            "evaluator_version", "development-quality-evaluation/0.2"
        ),
    )
    with pytest.raises(ValueError, match="QUALITY_SOURCE_LOCK_EVALUATOR_VERSION_MISMATCH"):
        build_scientific_policy()


def test_historical_quality_lock_source_hash_mutation_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mutated_quality_lock(
        tmp_path,
        monkeypatch,
        lambda locked: locked.__setitem__("evaluator_source_sha256", H3),
    )
    with pytest.raises(ValueError, match="QUALITY_SOURCE_LOCK_VERSION_MISMATCH"):
        build_scientific_policy()


def test_historical_quality_lock_resealed_source_list_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(locked: dict) -> None:
        locked["evaluator_source_files"][0]["path"] = "mutated/source.py"
        locked["evaluator_source_sha256"] = ft.sha256_value(locked["evaluator_source_files"])

    _mutated_quality_lock(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="QUALITY_SOURCE_LOCK_VERSION_MISMATCH"):
        build_scientific_policy()


def test_quality_lock_artifact_mutation_changes_fixed_target_source_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _temp_repo(tmp_path)
    lock_path = repo / ft.QUALITY_LOCK_RELATIVE_PATH
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text('{"lock":1}\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add quality lock")
    first = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(ft, "ROOT", repo)
    monkeypatch.setattr(
        ft,
        "fixed_target_source_paths",
        lambda: (ft.QUALITY_LOCK_RELATIVE_PATH, "pyproject.toml"),
    )
    before = source_inventory_at_commit(first)["source_inventory_sha256"]
    lock_path.write_text('{"lock":2}\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "mutate quality lock")
    second = _git(repo, "rev-parse", "HEAD")
    after = source_inventory_at_commit(second)["source_inventory_sha256"]
    assert before != after


def test_scientific_parent_identity_is_exact() -> None:
    policy = build_scientific_policy()
    expected_parent = {
        "implementation_commit": HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
        "evaluator_version": ft.HISTORICAL_QUALITY_EVALUATOR_VERSION,
        "evaluator_source_sha256": ft._QUALITY_SOURCE_LOCK_SHA256,
        "dataset_hash": ft.HISTORICAL_QUALITY_DATASET_HASH,
        "ordered_train_task_ids": list(ft.HISTORICAL_ORDERED_TRAIN_TASK_IDS),
        "ordered_eval_task_ids": list(ft.HISTORICAL_ORDERED_EVAL_TASK_IDS),
    }
    assert policy["ordered_train_task_ids"] == expected_parent["ordered_train_task_ids"]
    assert policy["ordered_eval_task_ids"] == expected_parent["ordered_eval_task_ids"]
    assert policy["frozen_scientific_parent"] == expected_parent
    validate_scientific_policy(policy)
