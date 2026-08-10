"""Fixed CPU target, observation, runtime, source, and acceptance foundations."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
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
QUALITY_LOCK_PATH = ROOT / "docs/evaluations/data/a2_a3_a4_heldout_summary.json"

TARGET_CONTRACT_VERSION = "toy-quality-fixed-cpu-target/1.0"
TARGET_OBSERVATION_VERSION = "toy-quality-fixed-cpu-target-observation/1.0"
FIXED_RUNTIME_VERSION = "toy-quality-canonical-cpu-runtime/1.1"
FIXED_TARGET_ACCEPTANCE_VERSION = "toy-quality-fixed-target-acceptance/1.0"
SOURCE_INVENTORY_VERSION = "toy-quality-fixed-target-source-inventory/1.0"
ATTEMPT_MANIFEST_VERSION = "toy-quality-fixed-target-attempt-manifest/1.0"

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

_FIXED_TARGET_SOURCE_ADDITIONS = {
    ".github/workflows/fixed-target-acceptance.yml",
    "planner_toy/__init__.py",
    "scripts/__init__.py",
    "planner_toy/schemas/toy_quality_fixed_cpu_target.schema.json",
    "planner_toy/schemas/toy_quality_fixed_runtime.schema.json",
    "planner_toy/schemas/toy_quality_fixed_target_acceptance.schema.json",
    "planner_toy/schemas/toy_quality_fixed_target_observation.schema.json",
    "requirements.lock",
    "pyproject.toml",
    "scripts/fixed_target_contract.py",
    "scripts/run_fixed_target_acceptance.py",
    # Bundle validation treats the retained probe as claim-bearing evidence.
    "scripts/canonical_training_probe_contract.py",
    "scripts/canonical_probe_evidence_validation.py",
    "scripts/canonical_cpu_hardware_fingerprint.py",
}
_QUALITY_SOURCE_LOCK_SHA256 = (
    "sha256:9205ad312fc37fa9927505e9c44a599e29fc5e31180db9d2e49ebfcc247b4570"
)

_TOP_LEVEL_EVALUATION_FILES = {
    "aggregate-summary.json",
    "canonical-semantic-payload.json",
    "dataset-manifest.json",
    "evaluation-config.json",
    "evaluation-manifest.json",
    "human-readable-examples.md",
    "paired-comparisons.json",
    "per-seed-summary.json",
    "replay-hash.txt",
    "task-results-semantic.json",
    "task-results.jsonl",
}
_TRAINING_RUN_FILES = {
    "checkpoint-manifest.json",
    "initialization.pt",
    "optimizer-evidence.json",
    "optimizer-state.pt",
    "trained.pt",
    "training-config.json",
    "training-report.json",
}


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


def _git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )


def _git_text(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def require_existing_commit(commit: str) -> str:
    _require_commit(commit, "FIXED_TARGET_SOURCE_COMMIT_INVALID")
    if _git_text("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise ValueError("FIXED_TARGET_SOURCE_COMMIT_NOT_FOUND")
    return commit


def require_trusted_implementation_commit(
    commit: str,
    *,
    protected_ref: str = "origin/main",
) -> str:
    require_existing_commit(commit)
    if _git_text("rev-parse", "--verify", f"{protected_ref}^{{commit}}").returncode:
        raise ValueError("FIXED_TARGET_PROTECTED_REF_NOT_FOUND")
    if _git_text("merge-base", "--is-ancestor", commit, protected_ref).returncode:
        raise ValueError("FIXED_TARGET_IMPLEMENTATION_NOT_TRUSTED")
    return commit


def _quality_source_paths() -> tuple[str, ...]:
    locked = json.loads(QUALITY_LOCK_PATH.read_text(encoding="utf-8"))
    entries = locked.get("evaluator_source_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("FIXED_TARGET_QUALITY_SOURCE_LOCK_INVALID")
    if locked.get("evaluator_source_sha256") != _QUALITY_SOURCE_LOCK_SHA256:
        raise ValueError("FIXED_TARGET_QUALITY_SOURCE_LOCK_VERSION_MISMATCH")
    if sha256_value(entries) != _QUALITY_SOURCE_LOCK_SHA256:
        raise ValueError("FIXED_TARGET_QUALITY_SOURCE_LOCK_HASH_MISMATCH")
    paths = tuple(entry.get("path") for entry in entries)
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("FIXED_TARGET_QUALITY_SOURCE_LOCK_INVALID")
    if len(paths) != len(set(paths)):
        raise ValueError("FIXED_TARGET_QUALITY_SOURCE_LOCK_DUPLICATE")
    return paths


def fixed_target_source_paths() -> tuple[str, ...]:
    return tuple(sorted(set(_quality_source_paths()) | _FIXED_TARGET_SOURCE_ADDITIONS))


def source_inventory_at_commit(commit: str) -> dict[str, Any]:
    require_existing_commit(commit)
    entries: list[dict[str, str]] = []
    for path in fixed_target_source_paths():
        process = _git_bytes("show", f"{commit}:{path}")
        if process.returncode:
            raise ValueError(f"FIXED_TARGET_SOURCE_MISSING:{path}")
        entries.append({"path": path, "sha256": sha256_bytes(process.stdout)})
    return {
        "source_inventory_version": SOURCE_INVENTORY_VERSION,
        "implementation_commit": commit,
        "files": entries,
        "source_inventory_sha256": sha256_value(entries),
    }


def validate_source_inventory(
    inventory: object,
    *,
    implementation_commit: str,
) -> dict[str, Any]:
    if not isinstance(inventory, dict) or set(inventory) != {
        "source_inventory_version",
        "implementation_commit",
        "files",
        "source_inventory_sha256",
    }:
        raise ValueError("FIXED_TARGET_SOURCE_INVENTORY_FIELDS_MISMATCH")
    if inventory["source_inventory_version"] != SOURCE_INVENTORY_VERSION:
        raise ValueError("FIXED_TARGET_SOURCE_INVENTORY_VERSION_MISMATCH")
    expected = source_inventory_at_commit(implementation_commit)
    if inventory != expected:
        raise ValueError("FIXED_TARGET_SOURCE_INVENTORY_MISMATCH")
    return inventory


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
    labels = contract["required_runner_labels"]
    if not REQUIRED_RUNNER_LABELS.issubset(set(labels)):
        raise ValueError("FIXED_TARGET_RUNNER_LABELS_MISMATCH")
    if labels != sorted(labels):
        raise ValueError("FIXED_TARGET_RUNNER_LABELS_NOT_CANONICAL")
    if contract["required_cpu_flags"] != sorted(contract["required_cpu_flags"]):
        raise ValueError("FIXED_TARGET_CPU_FLAGS_NOT_CANONICAL")
    forbidden = contract["forbidden_or_irrelevant_flags_policy"]["forbidden_flags"]
    if forbidden != sorted(forbidden):
        raise ValueError("FIXED_TARGET_FORBIDDEN_FLAGS_NOT_CANONICAL")
    if set(forbidden) & set(contract["required_cpu_flags"]):
        raise ValueError("FIXED_TARGET_CPU_FLAG_POLICY_CONTRADICTION")
    expected_dispatch = "DEFAULT" if contract["ATEN_CPU_CAPABILITY"] == "default" else "AVX2"
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
    runtime: dict[str, Any],
    target_contract: dict[str, Any] | None = None,
) -> None:
    _validate_schema(
        "toy_quality_fixed_runtime.schema.json",
        runtime,
        "FIXED_TARGET_RUNTIME_SCHEMA_INVALID",
    )
    if runtime["runtime_version"] != FIXED_RUNTIME_VERSION:
        raise ValueError("FIXED_TARGET_RUNTIME_MISMATCH")
    if target_contract is not None and runtime != build_runtime_contract(target_contract):
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
    contract: dict[str, Any],
    observation: dict[str, Any],
) -> None:
    validate_target_contract(contract)
    validate_target_observation(observation)
    if observation["target_contract_version"] != contract["target_contract_version"]:
        raise ValueError("FIXED_TARGET_RUNTIME_MISMATCH:target_contract_version")
    if observation["target_contract_sha256"] != target_contract_sha256(contract):
        raise ValueError("FIXED_TARGET_RUNTIME_MISMATCH:target_contract_sha256")

    for field in ("cpu_vendor", "cpu_family", "cpu_model", "cpu_stepping", "cpu_model_name"):
        if observation[field] != contract[field]:
            raise ValueError(f"FIXED_TARGET_CPU_MISMATCH:{field}")
    if not _microcode_satisfies(contract["microcode_policy"], observation["microcode"]):
        raise ValueError("FIXED_TARGET_CPU_MISMATCH:microcode")
    observed_flags = set(observation["cpu_flags"])
    if not set(contract["required_cpu_flags"]).issubset(observed_flags):
        raise ValueError("FIXED_TARGET_CPU_MISMATCH:required_cpu_flags")
    forbidden = set(contract["forbidden_or_irrelevant_flags_policy"]["forbidden_flags"])
    if forbidden & observed_flags:
        raise ValueError("FIXED_TARGET_CPU_MISMATCH:forbidden_cpu_flags")
    logical_policy = contract["logical_cpu_count_policy"]
    if (
        logical_policy["mode"] == "exact"
        and observation["logical_cpu_count"] != logical_policy["value"]
    ):
        raise ValueError("FIXED_TARGET_CPU_MISMATCH:logical_cpu_count")

    for field in (
        "os",
        "os_version",
        "architecture",
        "runner_type",
        "runner_image",
        "torch_num_threads",
        "torch_num_interop_threads",
        "mkldnn_enabled",
        "deterministic_algorithms",
        "deterministic_warn_only",
    ):
        if observation[field] != contract[field]:
            raise ValueError(f"FIXED_TARGET_RUNTIME_MISMATCH:{field}")
    if (
        contract["kernel_policy"]["mode"] == "exact"
        and observation["kernel_version"] != contract["kernel_policy"]["value"]
    ):
        raise ValueError("FIXED_TARGET_RUNTIME_MISMATCH:kernel_version")

    for field in (
        "python_implementation",
        "python_version",
        "python_build",
        "python_compiler",
        "pip_version",
        "torch_version",
        "torch_build_configuration_sha256",
        "mkl_available",
        "openmp_available",
        "mkldnn_available",
    ):
        if observation[field] != contract[field]:
            raise ValueError(f"FIXED_TARGET_SOFTWARE_MISMATCH:{field}")

    if observation["actual_atten_cpu_capability"] != contract["actual_atten_cpu_capability"]:
        raise ValueError("FIXED_TARGET_DISPATCH_MISMATCH")
    for field in (
        "ATEN_CPU_CAPABILITY",
        "MKL_CBWR",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if observation[field] != contract[field]:
            raise ValueError(f"FIXED_TARGET_ENV_MISMATCH:{field}")


def prepare_process_environment(contract: dict[str, Any]) -> None:
    validate_target_contract(contract)
    if "torch" in sys.modules:
        raise RuntimeError("FIXED_TARGET_TORCH_IMPORTED_BEFORE_PREFLIGHT")
    for name in (
        "ATEN_CPU_CAPABILITY",
        "MKL_CBWR",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
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
        contract["deterministic_algorithms"],
        warn_only=contract["deterministic_warn_only"],
    )
    torch.set_num_threads(contract["torch_num_threads"])
    try:
        torch.set_num_interop_threads(contract["torch_num_interop_threads"])
    except RuntimeError as error:
        raise RuntimeError("FIXED_TARGET_RUNTIME_CONFIGURATION_LATE") from error
    torch.backends.mkldnn.enabled = contract["mkldnn_enabled"]

    build_number, build_date = platform.python_build()
    observation: dict[str, Any] = {
        "observation_version": TARGET_OBSERVATION_VERSION,
        "target_contract_version": contract["target_contract_version"],
        "target_contract_sha256": target_contract_sha256(contract),
        "os": platform.system(),
        "os_version": _read_os_pretty_name(),
        "kernel_version": platform.release(),
        "architecture": platform.machine(),
        "cpu_vendor": cpu.get("vendor_id", ""),
        "cpu_family": cpu.get("cpu family", ""),
        "cpu_model": cpu.get("model", ""),
        "cpu_stepping": cpu.get("stepping", ""),
        "cpu_model_name": cpu.get("model name", ""),
        "microcode": cpu.get("microcode", ""),
        "cpu_flags": sorted(set(cpu.get("flags", "").split())),
        "logical_cpu_count": os.cpu_count(),
        "runner_type": "self-hosted-dedicated",
        "runner_image": os.environ.get("FIXED_TARGET_RUNNER_IMAGE", ""),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_build": [build_number, build_date],
        "python_compiler": platform.python_compiler(),
        "pip_version": importlib.metadata.version("pip"),
        "torch_version": torch.__version__,
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
        "deterministic_warn_only": (torch.is_deterministic_algorithms_warn_only_enabled()),
        "observation_sha256": "",
    }
    observation["observation_sha256"] = observation_sha256(observation)
    validate_observation_against_contract(contract, observation)
    return observation


def canonical_result_identity(claim_identities: dict[str, str]) -> str:
    if set(claim_identities) != set(CLAIM_IDENTITY_FIELDS):
        raise ValueError("FIXED_TARGET_CLAIM_IDENTITY_FIELDS_MISMATCH")
    for field in CLAIM_IDENTITY_FIELDS:
        _require_hash(
            claim_identities[field],
            f"FIXED_TARGET_CLAIM_IDENTITY_INVALID:{field}",
        )
    return sha256_value(claim_identities)


def acceptance_identity_sha256(acceptance: dict[str, Any]) -> str:
    return sha256_value(_payload_without_hash(acceptance, "acceptance_identity"))


def _validate_acceptance_record_structure(
    acceptance: dict[str, Any],
    *,
    allow_final_accepted: bool,
) -> None:
    _validate_schema(
        "toy_quality_fixed_target_acceptance.schema.json",
        acceptance,
        "FIXED_TARGET_ACCEPTANCE_SCHEMA_INVALID",
    )
    if acceptance["acceptance_version"] != FIXED_TARGET_ACCEPTANCE_VERSION:
        raise ValueError("FIXED_TARGET_ACCEPTANCE_VERSION_MISMATCH")
    observed_identity = _require_hash(
        acceptance["acceptance_identity"],
        "FIXED_TARGET_ACCEPTANCE_IDENTITY_INVALID",
    )
    if observed_identity != acceptance_identity_sha256(acceptance):
        raise ValueError("FIXED_TARGET_ACCEPTANCE_IDENTITY_MISMATCH")

    attempts = acceptance["attempts"]
    status = acceptance["status"]
    accepted = acceptance["accepted"]
    if status == "BLOCKED_ON_FIXED_RUNNER_PROVISIONING":
        if accepted or attempts:
            raise ValueError("FIXED_TARGET_BLOCKED_STATE_INVALID")
        if (
            acceptance["implementation_commit"] is not None
            or acceptance["target_contract"] is not None
            or acceptance["target_contract_sha256"] is not None
            or acceptance["runtime_contract"] is not None
            or acceptance["runtime_contract_sha256"] is not None
            or acceptance["blocker"] is None
            or acceptance["cross_attempt_comparison"]["status"] != "NOT_RUN"
        ):
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
        acceptance["implementation_commit"],
        "FIXED_TARGET_ACCEPTANCE_IMPLEMENTATION_INVALID",
    )
    indexes = [attempt["attempt_index"] for attempt in attempts]
    if indexes != list(range(1, len(attempts) + 1)):
        raise ValueError("FIXED_TARGET_ACCEPTANCE_ATTEMPT_ORDER_MISMATCH")
    workflow_ids = [attempt["workflow_run_id"] for attempt in attempts]
    job_ids = [attempt["job_id"] for attempt in attempts]
    if len(workflow_ids) != len(set(workflow_ids)):
        raise ValueError("FIXED_TARGET_ACCEPTANCE_DUPLICATE_WORKFLOW_RUN")
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("FIXED_TARGET_ACCEPTANCE_DUPLICATE_JOB")

    claim_blocks: list[dict[str, str]] = []
    probe_identities: list[str] = []
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
        if (
            attempt["checkpoint_origin_run_hash"] is not None
            or attempt["reuse_source_manifest_hash"] is not None
        ):
            raise ValueError("FIXED_TARGET_ACCEPTANCE_REUSED_LINEAGE")
        _require_hash(
            attempt["source_inventory_sha256"],
            "FIXED_TARGET_SOURCE_INVENTORY_HASH_INVALID",
        )
        _require_hash(
            attempt["probe_identity"],
            "FIXED_TARGET_PROBE_IDENTITY_INVALID",
        )
        if type(attempt["observed_optimizer_foreach"]) is not bool:
            raise ValueError("FIXED_TARGET_OPTIMIZER_OBSERVATION_INVALID:foreach")
        if type(attempt["observed_optimizer_fused"]) is not bool:
            raise ValueError("FIXED_TARGET_OPTIMIZER_OBSERVATION_INVALID:fused")
        claims = attempt["claim_identities"]
        computed = canonical_result_identity(claims)
        if attempt["canonical_result_identity"] != computed:
            raise ValueError("FIXED_TARGET_CANONICAL_RESULT_IDENTITY_MISMATCH")
        if attempt["result"] == "PASS" and not attempt["successful_full_evaluation"]:
            raise ValueError("FIXED_TARGET_ACCEPTANCE_FAILED_ATTEMPT")
        claim_blocks.append(claims)
        probe_identities.append(attempt["probe_identity"])

    exact_equal = (
        len(attempts) == 3
        and claim_blocks
        and all(claim == claim_blocks[0] for claim in claim_blocks)
        and len(set(probe_identities)) == 1
    )
    comparison = acceptance["cross_attempt_comparison"]
    if comparison["status"] == "PASS" and (not exact_equal or comparison["mismatches"]):
        raise ValueError("FIXED_TARGET_CROSS_ATTEMPT_COMPARISON_MISMATCH")
    if accepted:
        if not allow_final_accepted:
            raise ValueError("FIXED_TARGET_ACCEPTED_REQUIRES_BUNDLE")
        if status != "FIXED_TARGET_ACCEPTED":
            raise ValueError("FIXED_TARGET_ACCEPTED_STATUS_MISMATCH")
        if len(attempts) != 3:
            raise ValueError("FIXED_TARGET_ACCEPTANCE_ATTEMPT_COUNT_MISMATCH")
        if any(
            attempt["result"] != "PASS" or not attempt["successful_full_evaluation"]
            for attempt in attempts
        ):
            raise ValueError("FIXED_TARGET_ACCEPTANCE_FAILED_ATTEMPT")
        if comparison["status"] != "PASS" or not exact_equal:
            raise ValueError("FIXED_TARGET_CROSS_ATTEMPT_COMPARISON_MISMATCH")
        if acceptance["blocker"] is not None:
            raise ValueError("FIXED_TARGET_ACCEPTED_BLOCKER_PRESENT")
    elif status == "FIXED_TARGET_ACCEPTED":
        raise ValueError("FIXED_TARGET_ACCEPTED_STATUS_MISMATCH")


def validate_acceptance_record(acceptance: dict[str, Any]) -> None:
    """Validate a record envelope; final accepted evidence requires a bundle."""
    _validate_acceptance_record_structure(acceptance, allow_final_accepted=False)


# Compatibility name for blocked/provisional callers. It deliberately cannot accept final evidence.
validate_acceptance = validate_acceptance_record


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
    validate_acceptance_record(value)
    return value


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(code) from error
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _file_hash(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as error:
        raise ValueError(f"FIXED_TARGET_ATTEMPT_FILE_MISSING:{path.name}") from error


def _relative_file_hashes(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    exclude = exclude or set()
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("FIXED_TARGET_ATTEMPT_SYMLINK_FORBIDDEN")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in exclude:
                result[relative] = _file_hash(path)
    return result


def attempt_manifest_sha256(manifest: dict[str, Any]) -> str:
    return sha256_value(_payload_without_hash(manifest, "attempt_manifest_sha256"))


def validate_attempt_manifest(root: Path, manifest: dict[str, Any], index: int) -> None:
    required_fields = {
        "attempt_manifest_version",
        "attempt_index",
        "files",
        "attempt_manifest_sha256",
    }
    if set(manifest) != required_fields:
        raise ValueError("FIXED_TARGET_ATTEMPT_MANIFEST_FIELDS_MISMATCH")
    if manifest["attempt_manifest_version"] != ATTEMPT_MANIFEST_VERSION:
        raise ValueError("FIXED_TARGET_ATTEMPT_MANIFEST_VERSION_MISMATCH")
    if manifest["attempt_index"] != index:
        raise ValueError("FIXED_TARGET_ATTEMPT_MANIFEST_INDEX_MISMATCH")
    if not isinstance(manifest["files"], dict):
        raise ValueError("FIXED_TARGET_ATTEMPT_MANIFEST_FILES_INVALID")
    for path, digest in manifest["files"].items():
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            raise ValueError("FIXED_TARGET_ATTEMPT_MANIFEST_PATH_INVALID")
        _require_hash(digest, "FIXED_TARGET_ATTEMPT_MANIFEST_HASH_INVALID")
    observed_hash = _require_hash(
        manifest["attempt_manifest_sha256"],
        "FIXED_TARGET_ATTEMPT_MANIFEST_IDENTITY_INVALID",
    )
    if observed_hash != attempt_manifest_sha256(manifest):
        raise ValueError("FIXED_TARGET_ATTEMPT_MANIFEST_IDENTITY_MISMATCH")
    actual = _relative_file_hashes(root, exclude={"attempt_manifest.json"})
    if manifest["files"] != actual:
        raise ValueError("FIXED_TARGET_ATTEMPT_MANIFEST_COVERAGE_MISMATCH")


def _validate_attempt_shape(attempt_root: Path) -> None:
    if not attempt_root.is_dir():
        raise ValueError("FIXED_TARGET_ATTEMPT_DIRECTORY_MISSING")
    top = {path.name for path in attempt_root.iterdir()}
    if top != {"attempt_manifest.json", "preflight.json", "probe.json", "evaluation"}:
        raise ValueError("FIXED_TARGET_ATTEMPT_TOP_LEVEL_COVERAGE_MISMATCH")
    evaluation = attempt_root / "evaluation"
    top_files = {path.name for path in evaluation.iterdir() if path.is_file()}
    if top_files != _TOP_LEVEL_EVALUATION_FILES:
        raise ValueError("FIXED_TARGET_EVALUATION_TOP_LEVEL_COVERAGE_MISMATCH")
    if {path.name for path in evaluation.iterdir() if path.is_dir()} != {
        "training-runs",
        "evidence",
    }:
        raise ValueError("FIXED_TARGET_EVALUATION_DIRECTORY_COVERAGE_MISMATCH")
    run_dirs = sorted(evaluation.glob("training-runs/*/seed-*"))
    if len(run_dirs) != 9:
        raise ValueError("FIXED_TARGET_TRAINING_RUN_COVERAGE_MISMATCH")
    for run_dir in run_dirs:
        files = {path.name for path in run_dir.iterdir() if path.is_file()}
        if files != _TRAINING_RUN_FILES:
            raise ValueError("FIXED_TARGET_TRAINING_RUN_FILE_COVERAGE_MISMATCH")
        if any(path.is_dir() for path in run_dir.iterdir()):
            raise ValueError("FIXED_TARGET_TRAINING_RUN_EXTRA_DIRECTORY")


def _load_torch_object(path: Path) -> object:
    import torch  # noqa: PLC0415

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"FIXED_TARGET_TORCH_EVIDENCE_INVALID:{path.name}") from error


def _derive_optimizer_execution(evaluation_root: Path) -> tuple[bool, bool]:
    observed: set[tuple[object, object]] = set()
    for path in sorted(evaluation_root.glob("training-runs/*/seed-*/optimizer-state.pt")):
        state = _load_torch_object(path)
        if not isinstance(state, dict):
            raise ValueError("FIXED_TARGET_OPTIMIZER_STATE_INVALID")
        groups = state.get("param_groups")
        if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
            raise ValueError("FIXED_TARGET_OPTIMIZER_STATE_INVALID")
        observed.add((groups[0].get("foreach"), groups[0].get("fused")))
    if len(observed) != 1:
        raise ValueError("FIXED_TARGET_OPTIMIZER_EXECUTION_INCONSISTENT")
    foreach, fused = next(iter(observed))
    if foreach is not False or fused is not False:
        raise ValueError("FIXED_TARGET_OPTIMIZER_EXECUTION_MISMATCH")
    return foreach, fused


def _validate_evaluation_integrity(evaluation_root: Path) -> str:
    from planner_toy.quality import (  # noqa: PLC0415
        aggregate_summaries,
        canonical_replay_hash,
        canonical_replay_payload,
        paired,
        summarize,
        task_results_semantic_projection,
    )

    manifest = _read_json(
        evaluation_root / "evaluation-manifest.json",
        "FIXED_TARGET_EVALUATION_MANIFEST_INVALID",
    )
    artifact_files = _TOP_LEVEL_EVALUATION_FILES - {
        "evaluation-manifest.json",
        "replay-hash.txt",
    }
    if set(manifest.get("artifact_hashes", {})) != artifact_files:
        raise ValueError("FIXED_TARGET_EVALUATION_MANIFEST_COVERAGE_MISMATCH")
    for relative, digest in manifest["artifact_hashes"].items():
        if _file_hash(evaluation_root / relative) != digest:
            raise ValueError("FIXED_TARGET_EVALUATION_ARTIFACT_HASH_MISMATCH")

    checkpoint_files = {
        path.relative_to(evaluation_root).as_posix(): _file_hash(path)
        for path in sorted(evaluation_root.glob("training-runs/*/seed-*/*"))
        if path.is_file()
    }
    if manifest.get("checkpoint_manifest_hashes") != checkpoint_files:
        raise ValueError("FIXED_TARGET_CHECKPOINT_MANIFEST_COVERAGE_MISMATCH")
    evidence_files = {
        path.relative_to(evaluation_root).as_posix(): _file_hash(path)
        for path in sorted(evaluation_root.glob("evidence/*/seed-*/*/*"))
        if path.is_file()
    }
    if manifest.get("evidence_artifact_hashes") != evidence_files:
        raise ValueError("FIXED_TARGET_EVIDENCE_MANIFEST_COVERAGE_MISMATCH")

    config = _read_json(
        evaluation_root / "evaluation-config.json",
        "FIXED_TARGET_EVALUATION_CONFIG_INVALID",
    )
    if (
        config.get("training_execution_mode") != "TRAINED_IN_RUN"
        or config.get("checkpoint_origin_run_hash") is not None
        or config.get("reuse_source_manifest_hash") is not None
    ):
        raise ValueError("FIXED_TARGET_ACCEPTANCE_REUSED_LINEAGE")
    for path in sorted(evaluation_root.glob("training-runs/*/seed-*/checkpoint-manifest.json")):
        checkpoint = _read_json(path, "FIXED_TARGET_CHECKPOINT_MANIFEST_INVALID")
        if (
            checkpoint.get("training_execution_mode") != "TRAINED_IN_RUN"
            or checkpoint.get("checkpoint_origin_run_hash") is not None
            or checkpoint.get("reuse_source_manifest_hash") is not None
        ):
            raise ValueError("FIXED_TARGET_ACCEPTANCE_REUSED_LINEAGE")

    rows = [
        json.loads(line)
        for line in (evaluation_root / "task-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    semantic_rows = json.loads(
        (evaluation_root / "task-results-semantic.json").read_text(encoding="utf-8")
    )
    if semantic_rows != task_results_semantic_projection(rows):
        raise ValueError("FIXED_TARGET_TASK_RESULTS_SEMANTIC_MISMATCH")
    per_seed = json.loads((evaluation_root / "per-seed-summary.json").read_text(encoding="utf-8"))
    expected_per_seed = {
        f"{variant}/seed-{seed}": summarize(
            [row for row in rows if row["variant"] == variant and row["seed"] == seed]
        )
        for variant in config["variants"]
        for seed in config["seeds"]
    }
    if per_seed != expected_per_seed:
        raise ValueError("FIXED_TARGET_DERIVED_SUMMARY_MISMATCH")
    aggregate = json.loads((evaluation_root / "aggregate-summary.json").read_text(encoding="utf-8"))
    if aggregate != aggregate_summaries(
        expected_per_seed,
        rows,
        config["variants"],
        config["seeds"],
    ):
        raise ValueError("FIXED_TARGET_DERIVED_SUMMARY_MISMATCH")
    comparisons = json.loads(
        (evaluation_root / "paired-comparisons.json").read_text(encoding="utf-8")
    )
    expected_comparisons = [
        paired(rows, first, second)
        for first, second in (("A3", "A2"), ("A4", "A2"), ("A3", "A4"))
        if first in config["variants"] and second in config["variants"]
    ]
    if comparisons != expected_comparisons:
        raise ValueError("FIXED_TARGET_DERIVED_SUMMARY_MISMATCH")

    expected_payload = canonical_replay_payload(evaluation_root, manifest, rows)
    persisted_payload = json.loads(
        (evaluation_root / "canonical-semantic-payload.json").read_text(encoding="utf-8")
    )
    if persisted_payload != expected_payload:
        raise ValueError("FIXED_TARGET_CANONICAL_SEMANTIC_PAYLOAD_MISMATCH")
    replay = canonical_replay_hash(evaluation_root, manifest, rows)
    if (evaluation_root / "replay-hash.txt").read_text().strip() != replay:
        raise ValueError("FIXED_TARGET_REPLAY_HASH_MISMATCH")
    return replay


def _derive_claim_identities(evaluation_root: Path) -> dict[str, str]:
    from planner_toy.numeric_identity import (  # noqa: PLC0415
        canonical_state_dict_sha256,
        canonical_torch_object_sha256,
        exact_torch_object_sha256,
    )
    from planner_toy.training import state_dict_sha256  # noqa: PLC0415

    replay = _validate_evaluation_integrity(evaluation_root)
    init_exact: dict[str, str] = {}
    checkpoint_exact: dict[str, str] = {}
    optimizer_exact: dict[str, str] = {}
    canonical_states: dict[str, dict[str, str]] = {}
    training_configs: dict[str, object] = {}

    for run_dir in sorted(evaluation_root.glob("training-runs/*/seed-*")):
        relative = run_dir.relative_to(evaluation_root).as_posix()
        initialization = _load_torch_object(run_dir / "initialization.pt")
        trained = _load_torch_object(run_dir / "trained.pt")
        optimizer = _load_torch_object(run_dir / "optimizer-state.pt")
        checkpoint_manifest = _read_json(
            run_dir / "checkpoint-manifest.json",
            "FIXED_TARGET_CHECKPOINT_MANIFEST_INVALID",
        )
        initialization_identity = state_dict_sha256(initialization)
        trained_identity = state_dict_sha256(trained)
        optimizer_identity = exact_torch_object_sha256(optimizer)
        canonical_initialization = canonical_state_dict_sha256(initialization)
        canonical_trained = canonical_state_dict_sha256(trained)
        canonical_optimizer = canonical_torch_object_sha256(optimizer)
        expected_bindings = {
            "initialization_file_sha256": _file_hash(run_dir / "initialization.pt"),
            "initialization_state_dict_sha256": initialization_identity,
            "canonical_initialization_state_dict_sha256": canonical_initialization,
            "trained_file_sha256": _file_hash(run_dir / "trained.pt"),
            "trained_state_dict_sha256": trained_identity,
            "canonical_trained_state_dict_sha256": canonical_trained,
            "optimizer_state_file_sha256": _file_hash(run_dir / "optimizer-state.pt"),
            "optimizer_state_sha256": optimizer_identity,
            "canonical_optimizer_state_sha256": canonical_optimizer,
        }
        for field, expected in expected_bindings.items():
            if checkpoint_manifest.get(field) != expected:
                raise ValueError(f"FIXED_TARGET_CHECKPOINT_BINDING_MISMATCH:{field}")
        init_exact[relative] = initialization_identity
        checkpoint_exact[relative] = trained_identity
        optimizer_exact[relative] = optimizer_identity
        canonical_states[relative] = {
            "initialization": canonical_initialization,
            "trained": canonical_trained,
            "optimizer": canonical_optimizer,
        }
        training_configs[relative] = _read_json(
            run_dir / "training-config.json",
            "FIXED_TARGET_TRAINING_CONFIG_INVALID",
        )

    evaluation_config = _read_json(
        evaluation_root / "evaluation-config.json",
        "FIXED_TARGET_EVALUATION_CONFIG_INVALID",
    )
    task_results = json.loads(
        (evaluation_root / "task-results-semantic.json").read_text(encoding="utf-8")
    )
    canonical_payload = json.loads(
        (evaluation_root / "canonical-semantic-payload.json").read_text(encoding="utf-8")
    )
    summaries = {
        "per_seed": json.loads(
            (evaluation_root / "per-seed-summary.json").read_text(encoding="utf-8")
        ),
        "aggregate": json.loads(
            (evaluation_root / "aggregate-summary.json").read_text(encoding="utf-8")
        ),
        "paired": json.loads(
            (evaluation_root / "paired-comparisons.json").read_text(encoding="utf-8")
        ),
        "human_readable": (evaluation_root / "human-readable-examples.md").read_text(
            encoding="utf-8"
        ),
    }
    ordered_tasks = {
        "train_task_ids": evaluation_config.get("train_task_ids"),
        "eval_task_ids": evaluation_config.get("eval_task_ids"),
    }
    return {
        "initialization_identities_sha256": sha256_value(init_exact),
        "training_config_sha256": sha256_value(training_configs),
        "ordered_tasks_sha256": sha256_value(ordered_tasks),
        "checkpoint_identities_sha256": sha256_value(checkpoint_exact),
        "optimizer_state_identities_sha256": sha256_value(optimizer_exact),
        "canonical_state_dict_identities_sha256": sha256_value(canonical_states),
        "evaluation_task_results_sha256": sha256_value(task_results),
        "replay_hash": _require_hash(
            replay,
            "FIXED_TARGET_REPLAY_HASH_INVALID",
        ),
        "canonical_semantic_payload_sha256": sha256_value(canonical_payload),
        "derived_summaries_sha256": sha256_value(summaries),
    }


def _derive_probe_identity(probe_path: Path) -> str:
    from scripts.canonical_probe_evidence_validation import (  # noqa: PLC0415
        validate_probe_artifact,
    )
    from scripts.canonical_training_probe_contract import (  # noqa: PLC0415
        compute_probe_identity,
    )

    probe = _read_json(probe_path, "FIXED_TARGET_PROBE_INVALID")
    validate_probe_artifact(probe)
    identity = compute_probe_identity(probe)
    if probe["probe_identity"] != identity:
        raise ValueError("FIXED_TARGET_PROBE_IDENTITY_MISMATCH")
    return identity


def _validate_preflight(
    preflight: dict[str, Any],
    *,
    acceptance: dict[str, Any],
    attempt: dict[str, Any],
) -> None:
    required = {
        "implementation_commit",
        "target_contract",
        "target_contract_sha256",
        "runtime_contract",
        "runtime_contract_sha256",
        "target_observation",
        "source_inventory",
    }
    if set(preflight) != required:
        raise ValueError("FIXED_TARGET_PREFLIGHT_FIELDS_MISMATCH")
    if preflight["implementation_commit"] != acceptance["implementation_commit"]:
        raise ValueError("FIXED_TARGET_PREFLIGHT_IMPLEMENTATION_MISMATCH")
    if preflight["target_contract"] != acceptance["target_contract"]:
        raise ValueError("FIXED_TARGET_PREFLIGHT_TARGET_MISMATCH")
    if preflight["target_contract_sha256"] != acceptance["target_contract_sha256"]:
        raise ValueError("FIXED_TARGET_PREFLIGHT_TARGET_MISMATCH")
    if preflight["runtime_contract"] != acceptance["runtime_contract"]:
        raise ValueError("FIXED_TARGET_PREFLIGHT_RUNTIME_MISMATCH")
    if preflight["runtime_contract_sha256"] != acceptance["runtime_contract_sha256"]:
        raise ValueError("FIXED_TARGET_PREFLIGHT_RUNTIME_MISMATCH")
    if preflight["target_observation"] != attempt["target_observation"]:
        raise ValueError("FIXED_TARGET_PREFLIGHT_OBSERVATION_MISMATCH")
    inventory = validate_source_inventory(
        preflight["source_inventory"],
        implementation_commit=acceptance["implementation_commit"],
    )
    if inventory["source_inventory_sha256"] != attempt["source_inventory_sha256"]:
        raise ValueError("FIXED_TARGET_SOURCE_INVENTORY_MISMATCH")


def validate_acceptance_bundle(root: Path) -> dict[str, Any]:
    """Validate real 3/3 evidence. This is the only validator that may accept."""
    root = Path(root)
    acceptance = _read_json(
        root / "acceptance.json",
        "FIXED_TARGET_ACCEPTANCE_FILE_INVALID",
    )
    _validate_acceptance_record_structure(acceptance, allow_final_accepted=True)
    if not acceptance["accepted"]:
        raise ValueError("FIXED_TARGET_BUNDLE_NOT_FINAL_ACCEPTED")
    implementation = require_trusted_implementation_commit(acceptance["implementation_commit"])

    expected_root = {
        "acceptance.json",
        "attempt-1",
        "attempt-2",
        "attempt-3",
    }
    if {path.name for path in root.iterdir()} != expected_root:
        raise ValueError("FIXED_TARGET_BUNDLE_ROOT_COVERAGE_MISMATCH")

    derived_claims: list[dict[str, str]] = []
    derived_probes: list[str] = []
    for index, attempt in enumerate(acceptance["attempts"], start=1):
        attempt_root = root / f"attempt-{index}"
        _validate_attempt_shape(attempt_root)
        manifest = _read_json(
            attempt_root / "attempt_manifest.json",
            "FIXED_TARGET_ATTEMPT_MANIFEST_INVALID",
        )
        validate_attempt_manifest(attempt_root, manifest, index)
        preflight = _read_json(
            attempt_root / "preflight.json",
            "FIXED_TARGET_PREFLIGHT_INVALID",
        )
        _validate_preflight(preflight, acceptance=acceptance, attempt=attempt)

        claims = _derive_claim_identities(attempt_root / "evaluation")
        if claims != attempt["claim_identities"]:
            raise ValueError("FIXED_TARGET_DERIVED_CLAIM_IDENTITY_MISMATCH")
        if canonical_result_identity(claims) != attempt["canonical_result_identity"]:
            raise ValueError("FIXED_TARGET_CANONICAL_RESULT_IDENTITY_MISMATCH")

        foreach, fused = _derive_optimizer_execution(attempt_root / "evaluation")
        if (
            foreach != acceptance["runtime_contract"]["optimizer_foreach"]
            or fused != acceptance["runtime_contract"]["optimizer_fused"]
            or foreach != attempt["observed_optimizer_foreach"]
            or fused != attempt["observed_optimizer_fused"]
        ):
            raise ValueError("FIXED_TARGET_OPTIMIZER_EXECUTION_MISMATCH")

        probe_identity = _derive_probe_identity(attempt_root / "probe.json")
        if probe_identity != attempt["probe_identity"]:
            raise ValueError("FIXED_TARGET_PROBE_IDENTITY_MISMATCH")
        derived_claims.append(claims)
        derived_probes.append(probe_identity)

    if not all(claim == derived_claims[0] for claim in derived_claims):
        raise ValueError("FIXED_TARGET_CROSS_ATTEMPT_COMPARISON_MISMATCH")
    if len(set(derived_probes)) != 1:
        raise ValueError("FIXED_TARGET_CROSS_ATTEMPT_COMPARISON_MISMATCH")
    return {
        "valid": True,
        "accepted": True,
        "status": acceptance["status"],
        "implementation_commit": implementation,
        "canonical_result_identity": canonical_result_identity(derived_claims[0]),
        "probe_identity": derived_probes[0],
    }
