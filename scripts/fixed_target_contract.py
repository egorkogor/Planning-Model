"""Versioned fixed CPU target, runtime/1.1, observation, and acceptance contracts."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "planner_toy" / "schemas"

TARGET_CONTRACT_VERSION = "toy-quality-fixed-cpu-target/1.0"
TARGET_OBSERVATION_VERSION = "toy-quality-fixed-cpu-target-observation/1.0"
FIXED_RUNTIME_VERSION = "toy-quality-canonical-cpu-runtime/1.1"
FIXED_TARGET_ACCEPTANCE_VERSION = "toy-quality-fixed-target-acceptance/1.0"

_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")

REQUIRED_RUNNER_LABELS = {
    "self-hosted",
    "linux",
    "x64",
    "planning-model-canonical-cpu-v1",
}

CLAIM_IDENTITY_FIELDS = (
    "initialization_identities_sha256",
    "training_config_sha256",
    "ordered_tasks_sha256",
    "checkpoint_identities_sha256",
    "optimizer_state_identities_sha256",
    "canonical_state_dict_identities_sha256",
    "evaluation_task_results_sha256",
    "replay_hash",
    "canonical_semantic_payload_sha256",
    "derived_summaries_sha256",
)

FIXED_TARGET_SOURCE_PATHS = (
    ".github/workflows/fixed-target-acceptance.yml",
    "planner_toy/canonical.py",
    "planner_toy/canonical_runtime.py",
    "planner_toy/dataset.py",
    "planner_toy/domain.py",
    "planner_toy/e2e.py",
    "planner_toy/model.py",
    "planner_toy/numeric_identity.py",
    "planner_toy/quality.py",
    "planner_toy/semantic.py",
    "planner_toy/training.py",
    "planner_toy/schemas/toy_quality_fixed_cpu_target.schema.json",
    "planner_toy/schemas/toy_quality_fixed_runtime.schema.json",
    "planner_toy/schemas/toy_quality_fixed_target_acceptance.schema.json",
    "planner_toy/schemas/toy_quality_fixed_target_observation.schema.json",
    "requirements.lock",
    "scripts/fixed_target_contract.py",
    "scripts/run_fixed_target_acceptance.py",
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_value(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _payload_without_hash(value: dict[str, Any], hash_field: str) -> dict[str, Any]:
    clone = copy.deepcopy(value)
    clone.pop(hash_field, None)
    return clone


def _schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))


def _schema_registry() -> Registry:
    registry = Registry()
    for path in sorted(SCHEMA_ROOT.glob("toy_quality_fixed_*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return registry


def _validate_schema(name: str, value: object, code: str) -> None:
    errors = sorted(
        Draft202012Validator(_schema(name), registry=_schema_registry()).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise ValueError(f"{code}:{location}:{errors[0].message}")


def _require_hash(value: object, code: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(code)
    return value


def _require_commit(value: object, code: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(code)
    return value


def target_contract_sha256(contract: dict[str, Any]) -> str:
    validate_target_contract(contract)
    return sha256_value(contract)


def validate_target_contract(contract: dict[str, Any]) -> None:
    _validate_schema(
        "toy_quality_fixed_cpu_target.schema.json",
        contract,
        "FIXED_TARGET_CONTRACT_SCHEMA_INVALID",
    )
    if contract["target_contract_version"] != TARGET_CONTRACT_VERSION:
        raise ValueError("FIXED_TARGET_CONTRACT_VERSION_MISMATCH")
    if not REQUIRED_RUNNER_LABELS.issubset(set(contract["runner_labels"])):
        raise ValueError("FIXED_TARGET_RUNNER_LABELS_MISMATCH")
    if contract["runner_labels"] != sorted(contract["runner_labels"]):
        raise ValueError("FIXED_TARGET_RUNNER_LABELS_NOT_CANONICAL")
    if contract["required_cpu_flags"] != sorted(contract["required_cpu_flags"]):
        raise ValueError("FIXED_TARGET_CPU_FLAGS_NOT_CANONICAL")
    forbidden = contract["forbidden_or_irrelevant_flags_policy"]["forbidden_flags"]
    if forbidden != sorted(forbidden):
        raise ValueError("FIXED_TARGET_FORBIDDEN_FLAGS_NOT_CANONICAL")
    if set(forbidden) & set(contract["required_cpu_flags"]):
        raise ValueError("FIXED_TARGET_CPU_FLAG_POLICY_CONTRADICTION")
    if contract["ATEN_CPU_CAPABILITY"] == "default":
        expected_dispatch = "DEFAULT"
    elif contract["ATEN_CPU_CAPABILITY"] == "avx2":
        expected_dispatch = "AVX2"
    else:
        raise ValueError("FIXED_TARGET_DISPATCH_MISMATCH")
    if contract["actual_atten_cpu_capability"] != expected_dispatch:
        raise ValueError("FIXED_TARGET_DISPATCH_MISMATCH")
    if contract["optimizer_foreach"] is not False or contract["optimizer_fused"] is not False:
        raise ValueError("FIXED_TARGET_OPTIMIZER_EXECUTION_MISMATCH")


def build_runtime_contract(target_contract: dict[str, Any]) -> dict[str, Any]:
    validate_target_contract(target_contract)
    return {
        "runtime_version": FIXED_RUNTIME_VERSION,
        "fixed_target_contract_version": target_contract["target_contract_version"],
        "fixed_target_contract_sha256": target_contract_sha256(target_contract),
        "ATEN_CPU_CAPABILITY": target_contract["ATEN_CPU_CAPABILITY"],
        "actual_atten_cpu_capability": target_contract["actual_atten_cpu_capability"],
        "MKL_CBWR": target_contract["MKL_CBWR"],
        "torch_num_threads": target_contract["torch_num_threads"],
        "torch_num_interop_threads": target_contract["torch_num_interop_threads"],
        "OMP_NUM_THREADS": target_contract["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": target_contract["MKL_NUM_THREADS"],
        "OPENBLAS_NUM_THREADS": target_contract["OPENBLAS_NUM_THREADS"],
        "NUMEXPR_NUM_THREADS": target_contract["NUMEXPR_NUM_THREADS"],
        "mkldnn_enabled": target_contract["mkldnn_enabled"],
        "deterministic_algorithms": target_contract["deterministic_algorithms"],
        "deterministic_warn_only": target_contract["deterministic_warn_only"],
        "optimizer_class": "AdamW",
        "optimizer_foreach": target_contract["optimizer_foreach"],
        "optimizer_fused": target_contract["optimizer_fused"],
    }


def validate_runtime_contract(
    runtime: dict[str, Any], target_contract: dict[str, Any] | None = None
) -> None:
    _validate_schema(
        "toy_quality_fixed_runtime.schema.json",
        runtime,
        "FIXED_TARGET_RUNTIME_SCHEMA_INVALID",
    )
    if runtime["runtime_version"] != FIXED_RUNTIME_VERSION:
        raise ValueError("FIXED_TARGET_RUNTIME_MISMATCH")
    if target_contract is not None:
        expected = build_runtime_contract(target_contract)
        if runtime != expected:
            raise ValueError("FIXED_TARGET_RUNTIME_MISMATCH")


def runtime_contract_sha256(runtime: dict[str, Any]) -> str:
    validate_runtime_contract(runtime)
    return sha256_value(runtime)


def observation_sha256(observation: dict[str, Any]) -> str:
    return sha256_value(_payload_without_hash(observation, "observation_sha256"))


def validate_target_observation(observation: dict[str, Any]) -> None:
    _validate_schema(
        "toy_quality_fixed_target_observation.schema.json",
        observation,
        "FIXED_TARGET_OBSERVATION_SCHEMA_INVALID",
    )
    if observation["observation_version"] != TARGET_OBSERVATION_VERSION:
        raise ValueError("FIXED_TARGET_OBSERVATION_VERSION_MISMATCH")
    _require_hash(
        observation["target_contract_sha256"],
        "FIXED_TARGET_OBSERVATION_TARGET_HASH_INVALID",
    )
    observed_hash = _require_hash(
        observation["observation_sha256"],
        "FIXED_TARGET_OBSERVATION_HASH_INVALID",
    )
    if observed_hash != observation_sha256(observation):
        raise ValueError("FIXED_TARGET_OBSERVATION_HASH_MISMATCH")


def _microcode_satisfies(policy: dict[str, Any], observed: str) -> bool:
    mode = policy["mode"]
    expected = policy["value"]
    if mode in {"exact", "observed-only-acceptance-invalid-on-change"}:
        return observed == expected
    if mode == "minimum":
        try:
            return int(observed, 0) >= int(expected, 0)
        except ValueError:
            return False
    return False


def validate_observation_against_contract(
    contract: dict[str, Any], observation: dict[str, Any]
) -> None:
    validate_target_contract(contract)
    validate_target_observation(observation)
    contract_hash = target_contract_sha256(contract)
    if observation["target_contract_version"] != contract["target_contract_version"]:
        raise ValueError("FIXED_TARGET_RUNTIME_MISMATCH:target_contract_version")
    if observation["target_contract_sha256"] != contract_hash:
        raise ValueError("FIXED_TARGET_RUNTIME_MISMATCH:target_contract_sha256")

    for field in ("cpu_vendor", "cpu_family", "cpu_model", "cpu_stepping", "cpu_model_name"):
        if observation[field] != contract[field]:
            raise ValueError(f"FIXED_TARGET_CPU_MISMATCH:{field}")
    if not _microcode_satisfies(contract["microcode_policy"], observation["microcode"]):
        raise ValueError("FIXED_TARGET_CPU_MISMATCH:microcode")
    required_flags = set(contract["required_cpu_flags"])
    observed_flags = set(observation["cpu_flags"])
    if not required_flags.issubset(observed_flags):
        raise ValueError("FIXED_TARGET_CPU_MISMATCH:required_cpu_flags")
    forbidden = set(contract["forbidden_or_irrelevant_flags_policy"]["forbidden_flags"])
    if forbidden & observed_flags:
        raise ValueError("FIXED_TARGET_CPU_MISMATCH:forbidden_cpu_flags")
    logical_policy = contract["logical_cpu_count_policy"]
    if logical_policy["mode"] == "exact" and observation["logical_cpu_count"] != logical_policy["value"]:
        raise ValueError("FIXED_TARGET_CPU_MISMATCH:logical_cpu_count")

    runtime_pairs = (
        "os", "os_version", "architecture", "runner_type", "runner_image",
        "torch_num_threads", "torch_num_interop_threads", "mkldnn_enabled",
        "deterministic_algorithms", "deterministic_warn_only",
        "optimizer_foreach", "optimizer_fused",
    )
    for field in runtime_pairs:
        if observation[field] != contract[field]:
            raise ValueError(f"FIXED_TARGET_RUNTIME_MISMATCH:{field}")
    if observation["runner_labels"] != contract["runner_labels"]:
        raise ValueError("FIXED_TARGET_RUNTIME_MISMATCH:runner_labels")
    if contract["kernel_policy"]["mode"] == "exact" and observation["kernel_version"] != contract["kernel_policy"]["value"]:
        raise ValueError("FIXED_TARGET_RUNTIME_MISMATCH:kernel_version")

    for field in (
        "python_implementation", "python_version", "python_build", "python_compiler",
        "torch_version", "torch_build_configuration_sha256", "mkl_available",
        "openmp_available", "mkldnn_available",
    ):
        if observation[field] != contract[field]:
            raise ValueError(f"FIXED_TARGET_SOFTWARE_MISMATCH:{field}")

    if observation["actual_atten_cpu_capability"] != contract["actual_atten_cpu_capability"]:
        raise ValueError("FIXED_TARGET_DISPATCH_MISMATCH")
    for field in (
        "ATEN_CPU_CAPABILITY", "MKL_CBWR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        if observation[field] != contract[field]:
            raise ValueError(f"FIXED_TARGET_ENV_MISMATCH:{field}")


def prepare_process_environment(contract: dict[str, Any]) -> None:
    validate_target_contract(contract)
    if "torch" in sys.modules:
        raise RuntimeError("FIXED_TARGET_TORCH_IMPORTED_BEFORE_PREFLIGHT")
    for name in (
        "ATEN_CPU_CAPABILITY", "MKL_CBWR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        expected = str(contract[name])
        current = os.environ.get(name)
        if current not in {None, expected}:
            raise RuntimeError(f"FIXED_TARGET_ENV_MISMATCH:{name}")
        os.environ[name] = expected


def _read_os_pretty_name() -> str:
    path = Path("/etc/os-release")
    if not path.is_file():
        return platform.platform()
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values.get("PRETTY_NAME") or values.get("NAME") or platform.platform()


def _cpuinfo() -> dict[str, str]:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        raise RuntimeError("FIXED_TARGET_CPUINFO_UNAVAILABLE")
    first = path.read_text(encoding="utf-8", errors="replace").split("\n\n", 1)[0]
    result: dict[str, str] = {}
    for line in first.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def collect_target_observation(contract: dict[str, Any]) -> dict[str, Any]:
    prepare_process_environment(contract)
    cpu = _cpuinfo()
    import torch  # noqa: PLC0415

    torch.use_deterministic_algorithms(
        contract["deterministic_algorithms"], warn_only=contract["deterministic_warn_only"]
    )
    torch.set_num_threads(contract["torch_num_threads"])
    try:
        torch.set_num_interop_threads(contract["torch_num_interop_threads"])
    except RuntimeError as error:
        raise RuntimeError("FIXED_TARGET_RUNTIME_CONFIGURATION_LATE") from error
    torch.backends.mkldnn.enabled = contract["mkldnn_enabled"]

    labels = sorted(
        item.strip()
        for item in os.environ.get("FIXED_TARGET_RUNNER_LABELS", "").split(",")
        if item.strip()
    )
    build_number, build_date = platform.python_build()
    observation: dict[str, Any] = {
        "observation_version": TARGET_OBSERVATION_VERSION,
        "target_contract_version": contract["target_contract_version"],
        "target_contract_sha256": target_contract_sha256(contract),
        "os": platform.system(), "os_version": _read_os_pretty_name(),
        "kernel_version": platform.release(), "architecture": platform.machine(),
        "cpu_vendor": cpu.get("vendor_id", ""), "cpu_family": cpu.get("cpu family", ""),
        "cpu_model": cpu.get("model", ""), "cpu_stepping": cpu.get("stepping", ""),
        "cpu_model_name": cpu.get("model name", ""), "microcode": cpu.get("microcode", ""),
        "cpu_flags": sorted(set(cpu.get("flags", "").split())),
        "logical_cpu_count": os.cpu_count(),
        "runner_type": "self-hosted-dedicated", "runner_labels": labels,
        "runner_image": os.environ.get("FIXED_TARGET_RUNNER_IMAGE", ""),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(), "python_build": [build_number, build_date],
        "python_compiler": platform.python_compiler(), "torch_version": torch.__version__,
        "torch_build_configuration_sha256": sha256_bytes(torch.__config__.show().encode("utf-8")),
        "mkl_available": torch.backends.mkl.is_available(),
        "openmp_available": torch.backends.openmp.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY"),
        "actual_atten_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "MKL_CBWR": os.environ.get("MKL_CBWR"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "optimizer_foreach": False, "optimizer_fused": False,
        "observation_sha256": "",
    }
    observation["observation_sha256"] = observation_sha256(observation)
    validate_observation_against_contract(contract, observation)
    return observation


def canonical_result_identity(claim_identities: dict[str, str]) -> str:
    if set(claim_identities) != set(CLAIM_IDENTITY_FIELDS):
        raise ValueError("FIXED_TARGET_CLAIM_IDENTITY_FIELDS_MISMATCH")
    for field in CLAIM_IDENTITY_FIELDS:
        _require_hash(claim_identities[field], f"FIXED_TARGET_CLAIM_IDENTITY_INVALID:{field}")
    return sha256_value(claim_identities)


def acceptance_identity_sha256(acceptance: dict[str, Any]) -> str:
    return sha256_value(_payload_without_hash(acceptance, "acceptance_identity"))


def validate_acceptance(acceptance: dict[str, Any]) -> None:
    _validate_schema(
        "toy_quality_fixed_target_acceptance.schema.json",
        acceptance,
        "FIXED_TARGET_ACCEPTANCE_SCHEMA_INVALID",
    )
    if acceptance["acceptance_version"] != FIXED_TARGET_ACCEPTANCE_VERSION:
        raise ValueError("FIXED_TARGET_ACCEPTANCE_VERSION_MISMATCH")
    observed_identity = _require_hash(
        acceptance["acceptance_identity"], "FIXED_TARGET_ACCEPTANCE_IDENTITY_INVALID"
    )
    if observed_identity != acceptance_identity_sha256(acceptance):
        raise ValueError("FIXED_TARGET_ACCEPTANCE_IDENTITY_MISMATCH")

    attempts = acceptance["attempts"]
    accepted = acceptance["accepted"]
    status = acceptance["status"]
    if status == "BLOCKED_ON_FIXED_RUNNER_PROVISIONING":
        if accepted or attempts:
            raise ValueError("FIXED_TARGET_BLOCKED_STATE_INVALID")
        if acceptance["target_contract"] is not None or acceptance["target_contract_sha256"] is not None:
            raise ValueError("FIXED_TARGET_BLOCKED_STATE_INVALID")
        if acceptance["runtime_contract"] is not None or acceptance["runtime_contract_sha256"] is not None:
            raise ValueError("FIXED_TARGET_BLOCKED_STATE_INVALID")
        if acceptance["blocker"] is None or acceptance["cross_attempt_comparison"]["status"] != "NOT_RUN":
            raise ValueError("FIXED_TARGET_BLOCKED_STATE_INVALID")
        return

    target_contract = acceptance["target_contract"]
    runtime_contract = acceptance["runtime_contract"]
    if target_contract is None or runtime_contract is None:
        raise ValueError("FIXED_TARGET_ACCEPTANCE_CONTRACT_MISSING")
    validate_target_contract(target_contract)
    target_hash = target_contract_sha256(target_contract)
    if acceptance["target_contract_sha256"] != target_hash:
        raise ValueError("FIXED_TARGET_ACCEPTANCE_TARGET_HASH_MISMATCH")
    validate_runtime_contract(runtime_contract, target_contract)
    runtime_hash = runtime_contract_sha256(runtime_contract)
    if acceptance["runtime_contract_sha256"] != runtime_hash:
        raise ValueError("FIXED_TARGET_ACCEPTANCE_RUNTIME_HASH_MISMATCH")

    implementation = _require_commit(
        acceptance["implementation_commit"], "FIXED_TARGET_ACCEPTANCE_IMPLEMENTATION_INVALID"
    )
    expected_indexes = list(range(1, len(attempts) + 1))
    if [attempt["attempt_index"] for attempt in attempts] != expected_indexes:
        raise ValueError("FIXED_TARGET_ACCEPTANCE_ATTEMPT_ORDER_MISMATCH")
    workflow_ids = [attempt["workflow_run_id"] for attempt in attempts]
    job_ids = [attempt["job_id"] for attempt in attempts]
    if len(workflow_ids) != len(set(workflow_ids)):
        raise ValueError("FIXED_TARGET_ACCEPTANCE_DUPLICATE_WORKFLOW_RUN")
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("FIXED_TARGET_ACCEPTANCE_DUPLICATE_JOB")

    canonical_identities: list[str] = []
    probe_identities: list[str] = []
    claim_blocks: list[dict[str, str]] = []
    for attempt in attempts:
        if attempt["implementation_commit"] != implementation:
            raise ValueError("FIXED_TARGET_ACCEPTANCE_IMPLEMENTATION_MISMATCH")
        if attempt["target_contract_sha256"] != target_hash:
            raise ValueError("FIXED_TARGET_ACCEPTANCE_TARGET_MISMATCH")
        if attempt["runtime_contract_sha256"] != runtime_hash:
            raise ValueError("FIXED_TARGET_ACCEPTANCE_RUNTIME_MISMATCH")
        observation = attempt["target_observation"]
        validate_observation_against_contract(target_contract, observation)
        if attempt["target_observation_sha256"] != observation["observation_sha256"]:
            raise ValueError("FIXED_TARGET_OBSERVATION_HASH_MISMATCH")
        if attempt["training_execution_mode"] != "TRAINED_IN_RUN":
            raise ValueError("FIXED_TARGET_ACCEPTANCE_REUSED_LINEAGE")
        if attempt["checkpoint_origin_run_hash"] is not None or attempt["reuse_source_manifest_hash"] is not None:
            raise ValueError("FIXED_TARGET_ACCEPTANCE_REUSED_LINEAGE")
        claims = attempt["claim_identities"]
        computed = canonical_result_identity(claims)
        if attempt["canonical_result_identity"] != computed:
            raise ValueError("FIXED_TARGET_CANONICAL_RESULT_IDENTITY_MISMATCH")
        canonical_identities.append(computed)
        probe_identities.append(attempt["probe_identity"])
        claim_blocks.append(claims)
        _require_hash(attempt["probe_identity"], "FIXED_TARGET_PROBE_IDENTITY_INVALID")
        if attempt["result"] == "PASS" and not attempt["successful_full_evaluation"]:
            raise ValueError("FIXED_TARGET_ACCEPTANCE_FAILED_ATTEMPT")

    comparison = acceptance["cross_attempt_comparison"]
    exact_equal = (
        len(attempts) == 3
        and len(set(canonical_identities)) == 1
        and len(set(probe_identities)) == 1
        and all(claim == claim_blocks[0] for claim in claim_blocks)
    )
    if comparison["status"] == "PASS" and (not exact_equal or comparison["mismatches"]):
        raise ValueError("FIXED_TARGET_CROSS_ATTEMPT_COMPARISON_MISMATCH")
    if accepted:
        if status != "FIXED_TARGET_ACCEPTED":
            raise ValueError("FIXED_TARGET_ACCEPTED_STATUS_MISMATCH")
        if len(attempts) != 3:
            raise ValueError("FIXED_TARGET_ACCEPTANCE_ATTEMPT_COUNT_MISMATCH")
        if any(attempt["result"] != "PASS" or not attempt["successful_full_evaluation"] for attempt in attempts):
            raise ValueError("FIXED_TARGET_ACCEPTANCE_FAILED_ATTEMPT")
        if comparison["status"] != "PASS" or not exact_equal:
            raise ValueError("FIXED_TARGET_CROSS_ATTEMPT_COMPARISON_MISMATCH")
        if acceptance["blocker"] is not None:
            raise ValueError("FIXED_TARGET_ACCEPTED_BLOCKER_PRESENT")
    elif status == "FIXED_TARGET_ACCEPTED":
        raise ValueError("FIXED_TARGET_ACCEPTED_STATUS_MISMATCH")


def blocked_acceptance_record(required_action: str) -> dict[str, Any]:
    if not isinstance(required_action, str) or not required_action.strip():
        raise ValueError("FIXED_TARGET_BLOCKER_ACTION_REQUIRED")
    value: dict[str, Any] = {
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
            "required_action": required_action.strip(),
        },
        "acceptance_identity": "",
    }
    value["acceptance_identity"] = acceptance_identity_sha256(value)
    validate_acceptance(value)
    return value


def source_inventory_at_commit(commit: str) -> dict[str, Any]:
    _require_commit(commit, "FIXED_TARGET_SOURCE_COMMIT_INVALID")
    entries: list[dict[str, str]] = []
    for path in FIXED_TARGET_SOURCE_PATHS:
        process = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if process.returncode:
            raise ValueError(f"FIXED_TARGET_SOURCE_MISSING:{path}")
        entries.append({"path": path, "sha256": sha256_bytes(process.stdout)})
    entries.sort(key=lambda entry: entry["path"])
    return {
        "source_inventory_version": "toy-quality-fixed-target-source-inventory/1.0",
        "implementation_commit": commit,
        "files": entries,
        "source_inventory_sha256": sha256_value(entries),
    }
