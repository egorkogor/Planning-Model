from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

from planner_toy.canonical_runtime import (
    CANONICAL_CPU_RUNTIME_VERSION,
    FIXED_TARGET_POLICY_VERSION,
    HARDWARE_FINGERPRINT_VERSION,
    full_hardware_runtime_fingerprint,
    validate_fixed_execution_target,
    validate_supported_cpu_software_path,
)

PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def run_isolated(source: str, *, environment: dict[str, str] | None = None):
    result = subprocess.run(
        [sys.executable, "-c", source],
        env={**os.environ, **PINNED_ENV, **(environment or {})},
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_canonical_cpu_runtime_fingerprint_and_idempotence() -> None:
    fingerprint = run_isolated(
        """
import json
from planner_toy.canonical_runtime import configure_canonical_cpu_runtime
first = configure_canonical_cpu_runtime(17)
second = configure_canonical_cpu_runtime(17)
assert first == second
print(json.dumps(second, sort_keys=True))
"""
    )
    assert fingerprint == {
        "profile_version": "toy-quality-canonical-cpu-runtime/1.0",
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only_enabled": False,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "mkldnn_enabled": False,
        **PINNED_ENV,
    }


def test_incompatible_environment_fails_closed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from planner_toy.canonical_runtime import "
            "configure_canonical_cpu_runtime; configure_canonical_cpu_runtime()",
        ],
        env={**os.environ, **PINNED_ENV, "OMP_NUM_THREADS": "2"},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "CANONICAL_CPU_RUNTIME_ENV_MISMATCH:OMP_NUM_THREADS" in result.stderr


def test_late_interop_configuration_fails_closed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; torch.set_num_interop_threads(2); "
            "from planner_toy.canonical_runtime import "
            "configure_canonical_cpu_runtime; configure_canonical_cpu_runtime()",
        ],
        env={**os.environ, **PINNED_ENV},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "CANONICAL_CPU_RUNTIME_CONFIGURATION_LATE" in result.stderr


def test_runtime_1_0_is_not_reinterpreted_as_new_profile() -> None:
    assert CANONICAL_CPU_RUNTIME_VERSION == "toy-quality-canonical-cpu-runtime/1.0"
    assert HARDWARE_FINGERPRINT_VERSION == "toy-quality-cpu-hardware-fingerprint/1.0"
    assert FIXED_TARGET_POLICY_VERSION == "toy-quality-fixed-cpu-target-policy/1.0"


def test_full_hardware_fingerprint_has_required_stable_fields() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
    assert fingerprint["fingerprint_version"] == HARDWARE_FINGERPRINT_VERSION
    assert fingerprint["observation_identity_sha256"].startswith("sha256:")
    observed = fingerprint["observed_runtime_and_hardware"]
    assert set(observed) == {
        "fingerprint_version",
        "os",
        "cpu",
        "runner",
        "python",
        "pytorch",
        "canonical_runtime",
        "execution_environment",
    }
    assert {
        "vendor",
        "family",
        "model",
        "stepping",
        "model_name",
        "microcode",
        "logical_cpu_count",
        "flags_sha256",
        "capabilities",
    } <= set(observed["cpu"])
    assert {
        "version",
        "build_configuration",
        "build_configuration_sha256",
        "cpu_dispatch_capability",
        "mkl_available",
        "openmp_available",
        "mkldnn_available",
        "mkldnn_enabled",
    } <= set(observed["pytorch"])


def test_fingerprint_excludes_unstable_identity_fields() -> None:
    serialized = json.dumps(full_hardware_runtime_fingerprint(), sort_keys=True).lower()
    for forbidden in (
        "hostname",
        "host_name",
        "process_id",
        '"pid"',
        "timestamp",
        "current_time",
    ):
        assert forbidden not in serialized


def test_fixed_target_mismatch_fails_closed() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
    with pytest.raises(
        RuntimeError,
        match=r"CANONICAL_FIXED_TARGET_MISMATCH:cpu\.vendor",
    ):
        validate_fixed_execution_target(
            fingerprint,
            {"cpu.vendor": "definitely-not-this-cpu"},
        )


def test_unsupported_dispatch_path_fails_closed() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
    observed = fingerprint["observed_runtime_and_hardware"]
    actual = observed["pytorch"]["cpu_dispatch_capability"]
    impossible = "AVX512" if actual != "AVX512" else "DEFAULT"
    with pytest.raises(
        RuntimeError,
        match="CANONICAL_CPU_SOFTWARE_PATH_UNSUPPORTED:"
        "cpu_dispatch_capability",
    ):
        validate_supported_cpu_software_path(
            fingerprint,
            expected_dispatch=impossible,
        )


def test_unsupported_blas_path_fails_closed() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
    mutated = copy.deepcopy(fingerprint)
    mutated["observed_runtime_and_hardware"]["pytorch"]["mkl_available"] = False
    with pytest.raises(
        RuntimeError,
        match="CANONICAL_CPU_SOFTWARE_PATH_UNSUPPORTED:mkl_available",
    ):
        validate_supported_cpu_software_path(
            mutated,
            expected_dispatch=mutated["observed_runtime_and_hardware"]["pytorch"][
                "cpu_dispatch_capability"
            ],
        )
