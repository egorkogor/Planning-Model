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


def _observation_hash(observation: dict) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            observation,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def test_hardware_fingerprint_contains_required_stable_fields() -> None:
    fingerprint = full_hardware_runtime_fingerprint(
        {"profile_version": "toy-quality-canonical-cpu-runtime/1.0"}
    )
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


def test_hardware_fingerprint_excludes_unstable_identity_fields() -> None:
    serialized = json.dumps(
        full_hardware_runtime_fingerprint(),
        sort_keys=True,
    ).lower()
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
    fingerprint = full_hardware_runtime_fingerprint()
    del fingerprint[field]
    with pytest.raises(ValueError, match="TOP_LEVEL_FIELDS_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_extra_top_level_field() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
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
    fingerprint = full_hardware_runtime_fingerprint()
    observation = fingerprint["observed_runtime_and_hardware"]
    del observation[field]
    fingerprint["observation_identity_sha256"] = _observation_hash(observation)
    with pytest.raises(ValueError, match="OBSERVATION_FIELDS_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_extra_observation_field_after_hash_recompute() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
    observation = fingerprint["observed_runtime_and_hardware"]
    observation["accepted_target"] = True
    fingerprint["observation_identity_sha256"] = _observation_hash(observation)
    with pytest.raises(ValueError, match="OBSERVATION_FIELDS_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_version_mutation() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
    fingerprint["fingerprint_version"] = "mutated"
    with pytest.raises(ValueError, match="VERSION_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_nested_version_mutation_after_hash_recompute() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
    observation = fingerprint["observed_runtime_and_hardware"]
    observation["fingerprint_version"] = "mutated"
    fingerprint["observation_identity_sha256"] = _observation_hash(observation)
    with pytest.raises(ValueError, match="OBSERVATION_VERSION_MISMATCH"):
        validate_hardware_runtime_fingerprint(fingerprint)


def test_fingerprint_rejects_observation_mutation_with_stale_hash() -> None:
    fingerprint = full_hardware_runtime_fingerprint()
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
def test_fingerprint_rejects_noncanonical_hash_format(
    invalid_hash: str,
) -> None:
    fingerprint = full_hardware_runtime_fingerprint()
    fingerprint["observation_identity_sha256"] = invalid_hash
    with pytest.raises(ValueError, match="HASH_FORMAT_INVALID"):
        validate_hardware_runtime_fingerprint(fingerprint)
