"""Stable CPU/runtime evidence for canonical execution investigations.

This module observes investigation hosts without declaring any host an accepted
canonical execution target. It is intentionally outside the frozen
quality-v0.1 evaluator source inventory.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from pathlib import Path

import torch

HARDWARE_FINGERPRINT_VERSION = "toy-quality-cpu-hardware-fingerprint/1.0"
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TOP_LEVEL_FIELDS = {
    "fingerprint_version",
    "observation_identity_sha256",
    "observed_runtime_and_hardware",
}
_OBSERVATION_FIELDS = {
    "fingerprint_version",
    "os",
    "cpu",
    "runner",
    "python",
    "pytorch",
    "canonical_runtime",
    "execution_environment",
}
_OS_FIELDS = {"system", "release", "version", "machine", "architecture", "os_release"}
_CPU_FIELDS = {
    "vendor",
    "family",
    "model",
    "stepping",
    "model_name",
    "microcode",
    "logical_cpu_count",
    "flags_sha256",
    "capabilities",
}
_CPU_CAPABILITY_FIELDS = {
    "sse2",
    "avx",
    "avx2",
    "avx512f",
    "avx512dq",
    "avx512bw",
    "avx512vl",
    "fma",
}
_RUNNER_FIELDS = {
    "RUNNER_OS",
    "RUNNER_ARCH",
    "RUNNER_ENVIRONMENT",
    "ImageOS",
    "ImageVersion",
    "AZURE_REGION",
}
_PYTHON_FIELDS = {
    "implementation",
    "version",
    "compiler",
    "build_number",
    "build_date",
}
_PYTORCH_FIELDS = {
    "version",
    "cuda_version",
    "build_configuration",
    "build_configuration_sha256",
    "cpu_dispatch_capability",
    "mkl_available",
    "openmp_available",
    "mkldnn_available",
    "mkldnn_enabled",
}
_EXECUTION_ENVIRONMENT_FIELDS = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ATEN_CPU_CAPABILITY",
    "MKL_CBWR",
}
OBSERVED_EXECUTION_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ATEN_CPU_CAPABILITY",
    "MKL_CBWR",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_hash(value: object, error: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(error)
    return value


def _require_nonempty_string(value: object, error: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(error)
    return value


def _require_exact_fields(value: object, fields: set[str], error: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(error)
    return value


def _torch_build_configuration() -> str:
    return "\n".join(
        line.rstrip() for line in torch.__config__.show().strip().splitlines()
    )


def _python_build_fingerprint() -> dict[str, object]:
    build_number, build_date = platform.python_build()
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "compiler": platform.python_compiler(),
        "build_number": build_number,
        "build_date": build_date,
    }


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key] = value.strip().strip('"')
    return {key: values[key] for key in sorted(values)}


def _read_cpu_information() -> dict[str, object]:
    path = Path("/proc/cpuinfo")
    text = (
        path.read_text(encoding="utf-8", errors="replace")
        if path.is_file()
        else ""
    )
    first_block = text.split("\n\n", 1)[0]
    fields: dict[str, str] = {}
    for line in first_block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = " ".join(value.split())
    raw_flags = fields.get("flags") or fields.get("Features") or ""
    flags = sorted(set(raw_flags.split()))
    capability_names = (
        "sse2",
        "avx",
        "avx2",
        "avx512f",
        "avx512dq",
        "avx512bw",
        "avx512vl",
        "fma",
    )
    return {
        "vendor": fields.get("vendor_id") or fields.get("CPU implementer"),
        "family": fields.get("cpu family"),
        "model": fields.get("model"),
        "stepping": fields.get("stepping"),
        "model_name": fields.get("model name") or fields.get("Processor"),
        "microcode": fields.get("microcode"),
        "logical_cpu_count": os.cpu_count(),
        "flags_sha256": _sha256_bytes(
            "\n".join(flags).encode("ascii", errors="ignore")
        ),
        "capabilities": {name: name in flags for name in capability_names},
    }


def _runner_metadata() -> dict[str, object]:
    names = (
        "RUNNER_OS",
        "RUNNER_ARCH",
        "RUNNER_ENVIRONMENT",
        "ImageOS",
        "ImageVersion",
        "AZURE_REGION",
    )
    return {name: os.environ.get(name) for name in names}


def observed_runtime_and_hardware(
    runtime_fingerprint: dict[str, object],
) -> dict[str, object]:
    """Collect retained evidence without hostname, PID, or timestamps."""
    build_configuration = _torch_build_configuration()
    return {
        "fingerprint_version": HARDWARE_FINGERPRINT_VERSION,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
            "os_release": _read_os_release(),
        },
        "cpu": _read_cpu_information(),
        "runner": _runner_metadata(),
        "python": _python_build_fingerprint(),
        "pytorch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "build_configuration": build_configuration,
            "build_configuration_sha256": _sha256_bytes(
                build_configuration.encode("utf-8")
            ),
            "cpu_dispatch_capability": torch.backends.cpu.get_cpu_capability(),
            "mkl_available": torch.backends.mkl.is_available(),
            "openmp_available": torch.backends.openmp.is_available(),
            "mkldnn_available": torch.backends.mkldnn.is_available(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
        },
        "canonical_runtime": runtime_fingerprint,
        "execution_environment": {
            name: os.environ.get(name)
            for name in OBSERVED_EXECUTION_ENVIRONMENT
        },
    }


def full_hardware_runtime_fingerprint(
    runtime_fingerprint: dict[str, object],
) -> dict[str, object]:
    observation = observed_runtime_and_hardware(runtime_fingerprint)
    return {
        "fingerprint_version": HARDWARE_FINGERPRINT_VERSION,
        "observation_identity_sha256": _sha256_bytes(
            _canonical_bytes(observation)
        ),
        "observed_runtime_and_hardware": observation,
    }


def _validate_os(value: object) -> None:
    section = _require_exact_fields(
        value, _OS_FIELDS, "HARDWARE_FINGERPRINT_OS_FIELDS_MISMATCH"
    )
    for name in ("system", "release", "version", "machine", "architecture"):
        if not isinstance(section[name], str):
            raise ValueError(f"HARDWARE_FINGERPRINT_OS_FIELD_INVALID:{name}")
    os_release = section["os_release"]
    if not isinstance(os_release, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in os_release.items()
    ):
        raise ValueError("HARDWARE_FINGERPRINT_OS_RELEASE_INVALID")


def _validate_cpu(value: object) -> None:
    section = _require_exact_fields(
        value, _CPU_FIELDS, "HARDWARE_FINGERPRINT_CPU_FIELDS_MISMATCH"
    )
    for name in ("vendor", "family", "model", "stepping", "model_name", "microcode"):
        item = section[name]
        if item is not None and (not isinstance(item, str) or not item):
            raise ValueError(f"HARDWARE_FINGERPRINT_CPU_FIELD_INVALID:{name}")
    logical_cpu_count = section["logical_cpu_count"]
    if logical_cpu_count is not None and (
        type(logical_cpu_count) is not int or logical_cpu_count <= 0
    ):
        raise ValueError("HARDWARE_FINGERPRINT_CPU_LOGICAL_COUNT_INVALID")
    _require_hash(
        section["flags_sha256"], "HARDWARE_FINGERPRINT_CPU_FLAGS_HASH_INVALID"
    )
    capabilities = _require_exact_fields(
        section["capabilities"],
        _CPU_CAPABILITY_FIELDS,
        "HARDWARE_FINGERPRINT_CPU_CAPABILITIES_FIELDS_MISMATCH",
    )
    if not all(type(item) is bool for item in capabilities.values()):
        raise ValueError("HARDWARE_FINGERPRINT_CPU_CAPABILITY_VALUE_INVALID")


def _validate_runner(value: object) -> None:
    section = _require_exact_fields(
        value, _RUNNER_FIELDS, "HARDWARE_FINGERPRINT_RUNNER_FIELDS_MISMATCH"
    )
    if not all(item is None or isinstance(item, str) for item in section.values()):
        raise ValueError("HARDWARE_FINGERPRINT_RUNNER_VALUE_INVALID")


def _validate_python(value: object) -> None:
    section = _require_exact_fields(
        value, _PYTHON_FIELDS, "HARDWARE_FINGERPRINT_PYTHON_FIELDS_MISMATCH"
    )
    for name, item in section.items():
        _require_nonempty_string(
            item, f"HARDWARE_FINGERPRINT_PYTHON_FIELD_INVALID:{name}"
        )


def _validate_pytorch(value: object) -> None:
    section = _require_exact_fields(
        value, _PYTORCH_FIELDS, "HARDWARE_FINGERPRINT_PYTORCH_FIELDS_MISMATCH"
    )
    _require_nonempty_string(
        section["version"], "HARDWARE_FINGERPRINT_PYTORCH_VERSION_INVALID"
    )
    cuda_version = section["cuda_version"]
    if cuda_version is not None and not isinstance(cuda_version, str):
        raise ValueError("HARDWARE_FINGERPRINT_PYTORCH_CUDA_VERSION_INVALID")
    build_configuration = section["build_configuration"]
    if not isinstance(build_configuration, str):
        raise ValueError("HARDWARE_FINGERPRINT_PYTORCH_BUILD_CONFIGURATION_INVALID")
    build_hash = _require_hash(
        section["build_configuration_sha256"],
        "HARDWARE_FINGERPRINT_PYTORCH_BUILD_CONFIGURATION_HASH_INVALID",
    )
    expected_hash = _sha256_bytes(build_configuration.encode("utf-8"))
    if build_hash != expected_hash:
        raise ValueError("HARDWARE_FINGERPRINT_PYTORCH_BUILD_CONFIGURATION_HASH_MISMATCH")
    _require_nonempty_string(
        section["cpu_dispatch_capability"],
        "HARDWARE_FINGERPRINT_PYTORCH_DISPATCH_INVALID",
    )
    for name in (
        "mkl_available",
        "openmp_available",
        "mkldnn_available",
        "mkldnn_enabled",
    ):
        if type(section[name]) is not bool:
            raise ValueError(f"HARDWARE_FINGERPRINT_PYTORCH_BOOL_INVALID:{name}")


def _validate_execution_environment(value: object) -> None:
    section = _require_exact_fields(
        value,
        _EXECUTION_ENVIRONMENT_FIELDS,
        "HARDWARE_FINGERPRINT_EXECUTION_ENVIRONMENT_FIELDS_MISMATCH",
    )
    if not all(item is None or isinstance(item, str) for item in section.values()):
        raise ValueError("HARDWARE_FINGERPRINT_EXECUTION_ENVIRONMENT_VALUE_INVALID")


def validate_hardware_runtime_fingerprint(fingerprint: object) -> None:
    """Validate fingerprint structure and observation integrity only.

    Passing this validator does not mean that the observed host is an accepted
    canonical execution target.
    """
    if not isinstance(fingerprint, dict):
        raise ValueError("HARDWARE_FINGERPRINT_NOT_OBJECT")
    if set(fingerprint) != _TOP_LEVEL_FIELDS:
        raise ValueError("HARDWARE_FINGERPRINT_TOP_LEVEL_FIELDS_MISMATCH")
    if fingerprint["fingerprint_version"] != HARDWARE_FINGERPRINT_VERSION:
        raise ValueError("HARDWARE_FINGERPRINT_VERSION_MISMATCH")
    identity = fingerprint["observation_identity_sha256"]
    if not isinstance(identity, str) or _HASH_PATTERN.fullmatch(identity) is None:
        raise ValueError("HARDWARE_FINGERPRINT_HASH_FORMAT_INVALID")
    observation = fingerprint["observed_runtime_and_hardware"]
    if not isinstance(observation, dict):
        raise ValueError("HARDWARE_FINGERPRINT_OBSERVATION_NOT_OBJECT")
    if set(observation) != _OBSERVATION_FIELDS:
        raise ValueError("HARDWARE_FINGERPRINT_OBSERVATION_FIELDS_MISMATCH")
    if observation["fingerprint_version"] != HARDWARE_FINGERPRINT_VERSION:
        raise ValueError("HARDWARE_FINGERPRINT_OBSERVATION_VERSION_MISMATCH")

    _validate_os(observation["os"])
    _validate_cpu(observation["cpu"])
    _validate_runner(observation["runner"])
    _validate_python(observation["python"])
    _validate_pytorch(observation["pytorch"])
    from scripts.canonical_training_probe_contract import validate_runtime_fingerprint

    validate_runtime_fingerprint(observation["canonical_runtime"])
    _validate_execution_environment(observation["execution_environment"])

    expected = _sha256_bytes(_canonical_bytes(observation))
    if identity != expected:
        raise ValueError("HARDWARE_FINGERPRINT_OBSERVATION_HASH_MISMATCH")
