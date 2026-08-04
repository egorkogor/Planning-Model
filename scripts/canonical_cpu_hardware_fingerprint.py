"""Stable CPU/runtime evidence for canonical execution investigations.

This module is intentionally outside the frozen quality-v0.1 evaluator source
inventory. It observes candidate execution targets without reinterpreting the
historical ``toy-quality-canonical-cpu-runtime/1.0`` contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch

HARDWARE_FINGERPRINT_VERSION = "toy-quality-cpu-hardware-fingerprint/1.0"
FIXED_TARGET_POLICY_VERSION = "toy-quality-fixed-cpu-target-policy/1.0"
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
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
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
    runtime_fingerprint: dict[str, object] | None = None,
) -> dict[str, object]:
    """Collect deterministic evidence without hostname, PID, or timestamps."""
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
    runtime_fingerprint: dict[str, object] | None = None,
) -> dict[str, object]:
    observation = observed_runtime_and_hardware(runtime_fingerprint)
    return {
        "fingerprint_version": HARDWARE_FINGERPRINT_VERSION,
        "observation_identity_sha256": _sha256_bytes(_canonical_bytes(observation)),
        "observed_runtime_and_hardware": observation,
    }


def _lookup_path(value: dict[str, object], path: str) -> object:
    current: object = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise RuntimeError(f"CANONICAL_FIXED_TARGET_FIELD_MISSING:{path}")
        current = current[part]
    return current


def validate_fixed_execution_target(
    fingerprint: dict[str, object],
    expected_fields: dict[str, object],
) -> None:
    """Fail closed unless every versioned target field matches exactly."""
    if fingerprint.get("fingerprint_version") != HARDWARE_FINGERPRINT_VERSION:
        raise RuntimeError("CANONICAL_FIXED_TARGET_FINGERPRINT_VERSION_MISMATCH")
    observation = fingerprint.get("observed_runtime_and_hardware")
    if not isinstance(observation, dict):
        raise RuntimeError("CANONICAL_FIXED_TARGET_OBSERVATION_MISSING")
    for path, expected in sorted(expected_fields.items()):
        actual = _lookup_path(observation, path)
        if actual != expected:
            raise RuntimeError(
                "CANONICAL_FIXED_TARGET_MISMATCH:"
                f"{path}:expected={expected}:actual={actual}"
            )


def validate_supported_cpu_software_path(
    fingerprint: dict[str, object],
    *,
    expected_dispatch: str,
    require_mkl: bool = True,
    require_openmp: bool = True,
) -> None:
    observation = fingerprint.get("observed_runtime_and_hardware")
    if not isinstance(observation, dict):
        raise RuntimeError("CANONICAL_CPU_SOFTWARE_PATH_OBSERVATION_MISSING")
    pytorch = observation.get("pytorch")
    if not isinstance(pytorch, dict):
        raise RuntimeError("CANONICAL_CPU_SOFTWARE_PATH_PYTORCH_MISSING")
    checks: dict[str, Any] = {
        "cpu_dispatch_capability": expected_dispatch,
        "mkl_available": require_mkl,
        "openmp_available": require_openmp,
    }
    for name, expected in checks.items():
        actual = pytorch.get(name)
        if actual != expected:
            raise RuntimeError(
                "CANONICAL_CPU_SOFTWARE_PATH_UNSUPPORTED:"
                f"{name}:expected={expected}:actual={actual}"
            )
