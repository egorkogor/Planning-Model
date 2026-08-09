from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.fixed_target_contract import (
    CLAIM_IDENTITY_FIELDS,
    FIXED_TARGET_ACCEPTANCE_VERSION,
    TARGET_CONTRACT_VERSION,
    TARGET_OBSERVATION_VERSION,
    acceptance_identity_sha256,
    build_runtime_contract,
    canonical_result_identity,
    observation_sha256,
    runtime_contract_sha256,
    target_contract_sha256,
    validate_acceptance,
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
        "runner_labels": ["linux", "planning-model-canonical-cpu-v1", "self-hosted", "x64"],
        "runner_image": "planning-model-cpu-v1-image-sha256-0123456789abcdef",
        "python_implementation": "CPython",
        "python_version": "3.11.15",
        "python_build": ["main", "Aug 1 2026 00:00:00"],
        "python_compiler": "GCC 13.3.0",
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
        "runner_labels": contract["runner_labels"],
        "runner_image": contract["runner_image"],
        "python_implementation": contract["python_implementation"],
        "python_version": contract["python_version"],
        "python_build": contract["python_build"],
        "python_compiler": contract["python_compiler"],
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
        "optimizer_foreach": contract["optimizer_foreach"],
        "optimizer_fused": contract["optimizer_fused"],
        "observation_sha256": "",
    }
    observation["observation_sha256"] = observation_sha256(observation)
    return observation


def reseal_observation(observation: dict) -> dict:
    observation["observation_sha256"] = observation_sha256(observation)
    return observation


def claims(seed: str = "1") -> dict[str, str]:
    return {field: "sha256:" + seed * 64 for field in CLAIM_IDENTITY_FIELDS}


def valid_acceptance(*, attempts: int = 3, accepted: bool = True) -> dict:
    contract = valid_contract()
    runtime = build_runtime_contract(contract)
    target_hash = target_contract_sha256(contract)
    runtime_hash = runtime_contract_sha256(runtime)
    common_claims = claims()
    rows = []
    for index in range(1, attempts + 1):
        observation = valid_observation(contract)
        rows.append({
            "attempt_index": index,
            "workflow_run_id": 1000 + index,
            "job_id": 2000 + index,
            "implementation_commit": COMMIT,
            "target_contract_sha256": target_hash,
            "runtime_contract_sha256": runtime_hash,
            "target_observation": observation,
            "target_observation_sha256": observation["observation_sha256"],
            "probe_identity": H2,
            "canonical_result_identity": canonical_result_identity(common_claims),
            "claim_identities": copy.deepcopy(common_claims),
            "training_execution_mode": "TRAINED_IN_RUN",
            "checkpoint_origin_run_hash": None,
            "reuse_source_manifest_hash": None,
            "successful_full_evaluation": True,
            "result": "PASS",
        })
    value = {
        "acceptance_version": FIXED_TARGET_ACCEPTANCE_VERSION,
        "status": "FIXED_TARGET_ACCEPTED" if accepted else "TARGET_PROVISIONED",
        "implementation_commit": COMMIT,
        "target_contract": contract,
        "target_contract_sha256": target_hash,
        "runtime_contract": runtime,
        "runtime_contract_sha256": runtime_hash,
        "attempts": rows,
        "cross_attempt_comparison": {"status": "PASS" if accepted else "NOT_RUN", "mismatches": []},
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
        "blocker": {"code": "FIXED_RUNNER_NOT_PROVISIONED", "required_action": "Provision dedicated self-hosted Linux x86_64 runner."},
        "acceptance_identity": "",
    }
    value["acceptance_identity"] = acceptance_identity_sha256(value)
    return value


def reseal_acceptance(value: dict) -> dict:
    value["acceptance_identity"] = acceptance_identity_sha256(value)
    return value


def reseal_attempt_result(value: dict, index: int = 2) -> None:
    attempt = value["attempts"][index - 1]
    attempt["canonical_result_identity"] = canonical_result_identity(attempt["claim_identities"])
    reseal_acceptance(value)


def assert_rejected(callable_, code: str) -> None:
    with pytest.raises(ValueError, match=code):
        callable_()


def test_valid_contract_runtime_observation_and_acceptance() -> None:
    contract = valid_contract()
    validate_target_contract(contract)
    runtime = build_runtime_contract(contract)
    validate_runtime_contract(runtime, contract)
    observation = valid_observation(contract)
    validate_target_observation(observation)
    validate_observation_against_contract(contract, observation)
    validate_acceptance(valid_acceptance())
    validate_acceptance(blocked_acceptance())


@pytest.mark.parametrize(("field", "value", "code"), [
    ("cpu_model", "106", "FIXED_TARGET_CPU_MISMATCH:cpu_model"),
    ("cpu_stepping", "8", "FIXED_TARGET_CPU_MISMATCH:cpu_stepping"),
    ("actual_atten_cpu_capability", "DEFAULT", "FIXED_TARGET_DISPATCH_MISMATCH"),
    ("torch_build_configuration_sha256", H3, "FIXED_TARGET_SOFTWARE_MISMATCH:torch_build_configuration_sha256"),
    ("python_build", ["other", "Aug 2 2026"], "FIXED_TARGET_SOFTWARE_MISMATCH:python_build"),
    ("MKL_CBWR", "AUTO", "FIXED_TARGET_OBSERVATION_SCHEMA_INVALID"),
    ("OMP_NUM_THREADS", "2", "FIXED_TARGET_OBSERVATION_SCHEMA_INVALID"),
    ("mkldnn_enabled", True, "FIXED_TARGET_OBSERVATION_SCHEMA_INVALID"),
    ("optimizer_foreach", True, "FIXED_TARGET_OBSERVATION_SCHEMA_INVALID"),
    ("optimizer_fused", True, "FIXED_TARGET_OBSERVATION_SCHEMA_INVALID"),
])
def test_fully_resealed_observation_mismatches_rejected(field: str, value: object, code: str) -> None:
    contract = valid_contract()
    observation = valid_observation(contract)
    observation[field] = value
    reseal_observation(observation)
    assert_rejected(lambda: validate_observation_against_contract(contract, observation), code)


def test_required_cpu_flag_missing_rejected_after_reseal() -> None:
    contract = valid_contract()
    observation = valid_observation(contract)
    observation["cpu_flags"].remove("avx2")
    reseal_observation(observation)
    assert_rejected(lambda: validate_observation_against_contract(contract, observation), "FIXED_TARGET_CPU_MISMATCH:required_cpu_flags")


def test_mutated_observation_with_stale_hash_rejected() -> None:
    observation = valid_observation()
    observation["cpu_model"] = "106"
    assert_rejected(lambda: validate_target_observation(observation), "FIXED_TARGET_OBSERVATION_HASH_MISMATCH")


def test_fully_resealed_contradictory_observation_rejected() -> None:
    contract = valid_contract()
    observation = valid_observation(contract)
    observation["cpu_model"] = "106"
    reseal_observation(observation)
    assert_rejected(lambda: validate_observation_against_contract(contract, observation), "FIXED_TARGET_CPU_MISMATCH:cpu_model")


def test_extra_contract_field_rejected() -> None:
    contract = valid_contract(); contract["hostname"] = "must-not-be-semantic"
    assert_rejected(lambda: validate_target_contract(contract), "FIXED_TARGET_CONTRACT_SCHEMA_INVALID")


def test_missing_contract_field_rejected() -> None:
    contract = valid_contract(); contract.pop("cpu_model")
    assert_rejected(lambda: validate_target_contract(contract), "FIXED_TARGET_CONTRACT_SCHEMA_INVALID")


def test_wrong_runtime_version_rejected() -> None:
    runtime = build_runtime_contract(valid_contract()); runtime["runtime_version"] = "toy-quality-canonical-cpu-runtime/1.0"
    assert_rejected(lambda: validate_runtime_contract(runtime), "FIXED_TARGET_RUNTIME_SCHEMA_INVALID")


def test_runtime_1_0_cannot_masquerade_as_runtime_1_1() -> None:
    contract = valid_contract(); runtime = build_runtime_contract(contract); runtime["runtime_version"] = "toy-quality-canonical-cpu-runtime/1.0"
    assert runtime["fixed_target_contract_sha256"] == target_contract_sha256(contract)
    assert_rejected(lambda: validate_runtime_contract(runtime, contract), "FIXED_TARGET_RUNTIME_SCHEMA_INVALID")


def test_runtime_binds_exact_target_hash() -> None:
    contract = valid_contract(); runtime = build_runtime_contract(contract); runtime["fixed_target_contract_sha256"] = H3
    assert_rejected(lambda: validate_runtime_contract(runtime, contract), "FIXED_TARGET_RUNTIME_MISMATCH")


def test_contract_optimizer_path_is_explicit_false_false() -> None:
    contract = valid_contract(); contract["optimizer_foreach"] = True
    assert_rejected(lambda: validate_target_contract(contract), "FIXED_TARGET_CONTRACT_SCHEMA_INVALID")
    contract = valid_contract(); contract["optimizer_fused"] = True
    assert_rejected(lambda: validate_target_contract(contract), "FIXED_TARGET_CONTRACT_SCHEMA_INVALID")


def test_acceptance_with_only_two_attempts_cannot_be_accepted() -> None:
    value = valid_acceptance(attempts=2, accepted=True)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_SCHEMA_INVALID")


def test_three_attempts_different_implementation_sha_rejected_after_reseal() -> None:
    value = valid_acceptance(); value["attempts"][1]["implementation_commit"] = "b" * 40; reseal_acceptance(value)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_IMPLEMENTATION_MISMATCH")


def test_different_target_contract_rejected_after_reseal() -> None:
    value = valid_acceptance(); value["attempts"][1]["target_contract_sha256"] = H3; reseal_acceptance(value)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_TARGET_MISMATCH")


@pytest.mark.parametrize("claim_field", ["checkpoint_identities_sha256", "optimizer_state_identities_sha256", "replay_hash", "canonical_semantic_payload_sha256"])
def test_claim_bearing_difference_rejected_after_full_reseal(claim_field: str) -> None:
    value = valid_acceptance(); value["attempts"][1]["claim_identities"][claim_field] = H3; reseal_attempt_result(value, 2)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_CROSS_ATTEMPT_COMPARISON_MISMATCH")


def test_accepted_true_on_failed_attempt_rejected() -> None:
    value = valid_acceptance(); value["attempts"][1]["result"] = "FAIL"; value["attempts"][1]["successful_full_evaluation"] = False; reseal_acceptance(value)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_FAILED_ATTEMPT")


def test_reordered_attempts_rejected() -> None:
    value = valid_acceptance(); value["attempts"][0], value["attempts"][1] = value["attempts"][1], value["attempts"][0]; reseal_acceptance(value)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_ATTEMPT_ORDER_MISMATCH")


def test_duplicate_workflow_run_id_rejected() -> None:
    value = valid_acceptance(); value["attempts"][1]["workflow_run_id"] = value["attempts"][0]["workflow_run_id"]; reseal_acceptance(value)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_DUPLICATE_WORKFLOW_RUN")


def test_duplicate_job_id_rejected() -> None:
    value = valid_acceptance(); value["attempts"][1]["job_id"] = value["attempts"][0]["job_id"]; reseal_acceptance(value)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_DUPLICATE_JOB")


def test_reused_checkpoint_lineage_rejected_semantically() -> None:
    value = valid_acceptance(); value["attempts"][1]["training_execution_mode"] = "REUSED"; value["attempts"][1]["checkpoint_origin_run_hash"] = H3; reseal_acceptance(value)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_REUSED_LINEAGE")


def test_probe_identity_difference_rejected() -> None:
    value = valid_acceptance(); value["attempts"][1]["probe_identity"] = H3; reseal_acceptance(value)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_CROSS_ATTEMPT_COMPARISON_MISMATCH")


def test_acceptance_outer_stale_hash_rejected() -> None:
    value = valid_acceptance(); value["attempts"][1]["workflow_run_id"] = 9999
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_IDENTITY_MISMATCH")


def test_blocked_state_cannot_contain_fake_attempts() -> None:
    value = blocked_acceptance(); value["attempts"] = valid_acceptance()["attempts"][:1]; reseal_acceptance(value)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_SCHEMA_INVALID")


def test_target_provisioned_cannot_claim_accepted() -> None:
    value = valid_acceptance(accepted=False); value["accepted"] = True; reseal_acceptance(value)
    assert_rejected(lambda: validate_acceptance(value), "FIXED_TARGET_ACCEPTANCE_SCHEMA_INVALID")


def test_contract_disallows_ephemeral_identity_fields() -> None:
    for field in ("hostname", "pid", "job_id", "runner_ephemeral_id", "timestamp"):
        contract = valid_contract(); contract[field] = "x"
        assert_rejected(lambda contract=contract: validate_target_contract(contract), "FIXED_TARGET_CONTRACT_SCHEMA_INVALID")


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
    import subprocess
    for relative, expected in guards.items():
        completed = subprocess.run(["git", "hash-object", relative], cwd=root, text=True, capture_output=True, check=True)
        assert completed.stdout.strip() == expected
