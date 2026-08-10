from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

import scripts.fixed_target_contract as ft
from scripts.fixed_target_contract import (
    CLAIM_IDENTITY_FIELDS,
    FIXED_TARGET_ACCEPTANCE_VERSION,
    TARGET_CONTRACT_VERSION,
    TARGET_OBSERVATION_VERSION,
    acceptance_identity_sha256,
    attempt_manifest_sha256,
    build_runtime_contract,
    canonical_result_identity,
    fixed_target_source_paths,
    observation_sha256,
    require_trusted_implementation_commit,
    runtime_contract_sha256,
    source_inventory_at_commit,
    target_contract_sha256,
    validate_acceptance_bundle,
    validate_acceptance_record,
    validate_attempt_manifest,
    validate_observation_against_contract,
    validate_runtime_contract,
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
                "probe_identity": H2,
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
    with pytest.raises(ValueError, match="FIXED_TARGET_ACCEPTED_REQUIRES_BUNDLE"):
        validate_acceptance_record(valid_acceptance())


def test_provisional_record_still_checks_duplicate_run_ids() -> None:
    value = valid_acceptance(accepted=False)
    value["attempts"][1]["workflow_run_id"] = value["attempts"][0]["workflow_run_id"]
    reseal(value)
    with pytest.raises(ValueError, match="FIXED_TARGET_ACCEPTANCE_DUPLICATE_WORKFLOW_RUN"):
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
        (attempt / "probe.json").write_text("{}", encoding="utf-8")


def test_consistent_fake_claim_hashes_rejected_by_bundle_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance = valid_acceptance()
    _write_bundle_shell(tmp_path, acceptance)
    monkeypatch.setattr(ft, "require_existing_commit", lambda commit: commit)
    monkeypatch.setattr(ft, "_validate_attempt_shape", lambda path: None)
    monkeypatch.setattr(ft, "validate_attempt_manifest", lambda root, manifest, index: None)
    monkeypatch.setattr(ft, "_validate_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(ft, "_derive_optimizer_execution", lambda root: (False, False))
    monkeypatch.setattr(ft, "_derive_probe_identity", lambda path: H2)
    monkeypatch.setattr(ft, "_derive_claim_identities", lambda root: claims("3"))
    with pytest.raises(ValueError, match="FIXED_TARGET_DERIVED_CLAIM_IDENTITY_MISMATCH"):
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
    (attempt / "probe.json").write_bytes(b"{}")
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
        "docs/evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md": "e2344c07a76fcf7de140f894317fb509f6bc04fb",
        "docs/evaluations/data/a2_a3_a4_heldout_summary.json": "408742e15a3cddacdefcb0f0b814a6d68a5ca62d",
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
