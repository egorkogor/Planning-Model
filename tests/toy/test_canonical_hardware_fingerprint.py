from __future__ import annotations

import copy
import json

import pytest

from scripts.canonical_cpu_hardware_fingerprint import (
    FIXED_TARGET_POLICY_VERSION,
    HARDWARE_FINGERPRINT_VERSION,
    full_hardware_runtime_fingerprint,
    validate_fixed_execution_target,
    validate_supported_cpu_software_path,
)


def test_hardware_fingerprint_contains_required_stable_fields() -> None:
    fingerprint = full_hardware_runtime_fingerprint(
        {"profile_version": "toy-quality-canonical-cpu-runtime/1.0"}
    )
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


def test_hardware_fingerprint_excludes_unstable_identity_fields() -> None:
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


def test_fixed_execution_target_mismatch_fails_closed() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
    with pytest.raises(
        RuntimeError,
        match=r"CANONICAL_FIXED_TARGET_MISMATCH:cpu\.vendor",
    ):
        validate_fixed_execution_target(
            fingerprint,
            {"cpu.vendor": "not-the-observed-vendor"},
        )


def test_unsupported_dispatch_and_blas_paths_fail_closed() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
    observed = fingerprint["observed_runtime_and_hardware"]["pytorch"]
    impossible_dispatch = (
        "AVX512" if observed["cpu_dispatch_capability"] != "AVX512" else "DEFAULT"
    )
    with pytest.raises(
        RuntimeError,
        match="CANONICAL_CPU_SOFTWARE_PATH_UNSUPPORTED:cpu_dispatch_capability",
    ):
        validate_supported_cpu_software_path(
            fingerprint,
            expected_dispatch=impossible_dispatch,
        )
    mutated = copy.deepcopy(fingerprint)
    mutated["observed_runtime_and_hardware"]["pytorch"]["mkl_available"] = False
    with pytest.raises(
        RuntimeError,
        match="CANONICAL_CPU_SOFTWARE_PATH_UNSUPPORTED:mkl_available",
    ):
        validate_supported_cpu_software_path(
            mutated,
            expected_dispatch=observed["cpu_dispatch_capability"],
        )


def test_new_policy_does_not_reinterpret_runtime_1_0() -> None:
    assert FIXED_TARGET_POLICY_VERSION == "toy-quality-fixed-cpu-target-policy/1.0"
    fingerprint = full_hardware_runtime_fingerprint(
        {"profile_version": "toy-quality-canonical-cpu-runtime/1.0"}
    )
    assert (
        fingerprint["observed_runtime_and_hardware"]["canonical_runtime"]
        ["profile_version"]
        == "toy-quality-canonical-cpu-runtime/1.0"
    )
