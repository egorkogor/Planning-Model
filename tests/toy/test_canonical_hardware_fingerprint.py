from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.canonical_cpu_hardware_fingerprint import (
    HARDWARE_FINGERPRINT_VERSION,
    full_hardware_runtime_fingerprint,
    validate_hardware_runtime_fingerprint,
)


def _runtime() -> dict[str, object]:
    return {
        "profile_version": "toy-quality-canonical-cpu-runtime/1.0",
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only_enabled": False,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "mkldnn_enabled": False,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _fingerprint() -> dict:
    return full_hardware_runtime_fingerprint(_runtime())


def _observation_hash(observation: dict) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            observation,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _reseal_observation(fingerprint: dict) -> None:
    fingerprint["observation_identity_sha256"] = _observation_hash(
        fingerprint["observed_runtime_and_hardware"]
    )


def test_hardware_fingerprint_contains_required_stable_fields() -> None:
    fingerprint = _fingerprint()
    validate_hardware_runtime_fingerprint(fingerprint)
    assert fingerprint["fingerprint_version"] == HARDWARE_FINGERPRINT_VERSION
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
    assert set(observed["os"]) == {
        "system",
        "release",
        "version",
        "machine",
        "architecture",
        "os_release",
    }
    assert set(observed["cpu"]) == {
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
    assert set(observed["cpu"]["capabilities"]) == {
        "sse2",
        "avx",
        "avx2",
        "avx512f",
        "avx512dq",
        "avx512bw",
        "avx512vl",
        "fma",
    }
    assert set(observed["runner"]) == {
        "RUNNER_OS",
        "RUNNER_ARCH",
        "RUNNER_ENVIRONMENT",
        "ImageOS",
        "ImageVersion",
        "AZURE_REGION",
    }
    assert set(observed["python"]) == {
        "implementation",
        "version",
        "compiler",
        "build_number",
        "build_date",
    }
    assert set(observed["pytorch"]) == {
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
    assert set(observed["execution_environment"]) == {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ATEN_CPU_CAPABILITY",
        "MKL_CBWR",
    }
    assert observed["canonical_runtime"] == _runtime()


def test_hardware_fingerprint_excludes_unstable_identity_fields() -> None:
    serialized = json.dumps(_fingerprint(), sort_keys=True).lower()
    for forbidden in (
        "hostname",
        "host_name",
        "process_id",
        '"pid"',
        "timestamp",
        "current_time",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "field",
    sorted(
        {
            "fingerprint_version",
            "observation_identity_sha256",
            "observed_runtime_and_hardware",
        }
    ),
)
def test_fingerprint_rejects_missing_top_level_field(field: str) -> None:
    fingerprint = _fingerprint()
    del fingerprint[field]
    with pytest.raises(ValueError, match="TOP_LEVEL_FIELDS_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_extra_top_level_field() -> None:
    fingerprint = _fingerprint()
    fingerprint["accepted_target"] = True
    with pytest.raises(ValueError, match="TOP_LEVEL_FIELDS_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


@pytest.mark.parametrize(
    "field",
    sorted(
        {
            "fingerprint_version",
            "os",
            "cpu",
            "runner",
            "python",
            "pytorch",
            "canonical_runtime",
            "execution_environment",
        }
    ),
)
def test_fingerprint_rejects_missing_observation_field_after_hash_recompute(
    field: str,
) -> None:
    fingerprint = _fingerprint()
    observation = fingerprint["observed_runtime_and_hardware"]
    del observation[field]
    _reseal_observation(fingerprint)
    with pytest.raises(ValueError, match="OBSERVATION_FIELDS_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_extra_observation_field_after_hash_recompute() -> None:
    fingerprint = _fingerprint()
    observation = fingerprint["observed_runtime_and_hardware"]
    observation["accepted_target"] = True
    _reseal_observation(fingerprint)
    with pytest.raises(ValueError, match="OBSERVATION_FIELDS_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_version_mutation() -> None:
    fingerprint = _fingerprint()
    fingerprint["fingerprint_version"] = "mutated"
    with pytest.raises(ValueError, match="VERSION_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_nested_version_mutation_after_hash_recompute() -> None:
    fingerprint = _fingerprint()
    observation = fingerprint["observed_runtime_and_hardware"]
    observation["fingerprint_version"] = "mutated"
    _reseal_observation(fingerprint)
    with pytest.raises(ValueError, match="OBSERVATION_VERSION_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_observation_mutation_with_stale_hash() -> None:
    fingerprint = _fingerprint()
    mutated = copy.deepcopy(fingerprint)
    mutated["observed_runtime_and_hardware"]["cpu"]["vendor"] = "mutated"
    with pytest.raises(ValueError, match="OBSERVATION_HASH_MISMATCH"):
        validate_hardware_runtime_fingerprint(mutated)


@pytest.mark.parametrize(
    "invalid_hash",
    (
        "0" * 64,
        "sha256:" + "A" * 64,
        "sha256:" + "0" * 63,
        "sha512:" + "0" * 64,
    ),
)
def test_fingerprint_rejects_noncanonical_hash_format(invalid_hash: str) -> None:
    fingerprint = _fingerprint()
    fingerprint["observation_identity_sha256"] = invalid_hash
    with pytest.raises(ValueError, match="HASH_FORMAT_INVALID"):
        validate_hardware_runtime_fingerprint(fingerprint)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("cpu_missing_vendor", "CPU_FIELDS_MISMATCH"),
        ("cpu_extra_field", "CPU_FIELDS_MISMATCH"),
        ("capabilities_missing_avx2", "CPU_CAPABILITIES_FIELDS_MISMATCH"),
        ("python_missing_compiler", "PYTHON_FIELDS_MISMATCH"),
        ("pytorch_missing_dispatch", "PYTORCH_FIELDS_MISMATCH"),
        ("pytorch_build_hash_mismatch", "BUILD_CONFIGURATION_HASH_MISMATCH"),
        ("runner_extra_field", "RUNNER_FIELDS_MISMATCH"),
        ("environment_missing_mkl_cbwr", "EXECUTION_ENVIRONMENT_FIELDS_MISMATCH"),
    ),
)
def test_fingerprint_rejects_resealed_nested_schema_mutation(
    mutation: str, error: str
) -> None:
    fingerprint = _fingerprint()
    observed = fingerprint["observed_runtime_and_hardware"]
    if mutation == "cpu_missing_vendor":
        del observed["cpu"]["vendor"]
    elif mutation == "cpu_extra_field":
        observed["cpu"]["extra"] = "x"
    elif mutation == "capabilities_missing_avx2":
        del observed["cpu"]["capabilities"]["avx2"]
    elif mutation == "python_missing_compiler":
        del observed["python"]["compiler"]
    elif mutation == "pytorch_missing_dispatch":
        del observed["pytorch"]["cpu_dispatch_capability"]
    elif mutation == "pytorch_build_hash_mismatch":
        observed["pytorch"]["build_configuration_sha256"] = "sha256:" + "0" * 64
    elif mutation == "runner_extra_field":
        observed["runner"]["extra"] = None
    elif mutation == "environment_missing_mkl_cbwr":
        del observed["execution_environment"]["MKL_CBWR"]
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _reseal_observation(fingerprint)
    with pytest.raises(ValueError, match=error):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_non_runtime_canonical_runtime_after_reseal() -> None:
    fingerprint = _fingerprint()
    fingerprint["observed_runtime_and_hardware"]["canonical_runtime"] = None
    _reseal_observation(fingerprint)
    with pytest.raises(ValueError, match="PROBE_RUNTIME_NOT_OBJECT"):
        validate_hardware_runtime_fingerprint(fingerprint)
