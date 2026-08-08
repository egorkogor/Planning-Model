from __future__ import annotations

import copy
import importlib

import pytest

from scripts.canonical_probe_evidence_validation import (
    validate_probe_artifact as authoritative_validate_probe_artifact,
)
from scripts.canonical_training_probe_contract import (
    CANONICAL_RUNTIME_VERSION,
    compute_evidence_identity,
    compute_probe_identity,
    validate_execution_contract,
    validate_runtime_fingerprint,
)
from scripts.run_canonical_training_probe import (
    validate_probe_artifact as cli_validate_probe_artifact,
)
from tests.toy.test_canonical_training_probe_hardening import (
    _hash,
    _probe,
    _reseal_hardware,
    _runtime,
)

probe_contract = importlib.import_module("scripts.canonical_training_probe_contract")
probe_core = importlib.import_module("scripts.canonical_training_probe_core")

_PROBE_IDENTITY_FIELDS = (
    "probe_version",
    "variant",
    "seed",
    "epochs",
    "ordered_train_task_ids",
    "parameter_names",
    "initial_parameters",
    "updates",
    "execution_contract",
    "execution_contract_sha256",
)


def _reseal_probe_identity_without_validation(payload: dict) -> None:
    identity_payload = {key: payload[key] for key in _PROBE_IDENTITY_FIELDS}
    payload["probe_identity"] = _hash(identity_payload)


def test_only_one_authoritative_probe_artifact_validator() -> None:
    assert not hasattr(probe_contract, "validate_probe_artifact")
    assert not hasattr(probe_core, "validate_probe_artifact")
    assert cli_validate_probe_artifact is authoritative_validate_probe_artifact


def test_resealed_cross_binding_contradiction_fails_authoritative_and_core() -> None:
    valid_payload = _probe()
    payload = copy.deepcopy(valid_payload)
    payload["execution_contract"]["python_version"] = "3.13.7"
    observation = payload["hardware_runtime_fingerprint"][
        "observed_runtime_and_hardware"
    ]
    observation["python"]["version"] = "3.12.10"

    _reseal_hardware(payload)
    payload["execution_contract_sha256"] = _hash(payload["execution_contract"])
    payload["probe_identity"] = compute_probe_identity(payload)
    payload["evidence_identity"] = compute_evidence_identity(payload)

    with pytest.raises(
        ValueError, match="PROBE_RUNTIME_CROSS_BINDING_MISMATCH:python_version"
    ) as authoritative_error:
        authoritative_validate_probe_artifact(payload)
    with pytest.raises(
        ValueError, match="PROBE_RUNTIME_CROSS_BINDING_MISMATCH:python_version"
    ) as core_error:
        probe_core.compare_probes(payload, valid_payload)
    assert str(core_error.value) == str(authoritative_error.value)


def test_fully_resealed_runtime_1_1_artifact_is_rejected() -> None:
    payload = copy.deepcopy(_probe())
    unsupported_version = "toy-quality-canonical-cpu-runtime/1.1"

    payload["execution_contract"]["canonical_runtime_version"] = unsupported_version
    payload["runtime"]["profile_version"] = unsupported_version
    observation = payload["hardware_runtime_fingerprint"][
        "observed_runtime_and_hardware"
    ]
    observation["canonical_runtime"]["profile_version"] = unsupported_version

    payload["execution_contract_sha256"] = _hash(payload["execution_contract"])
    _reseal_probe_identity_without_validation(payload)
    _reseal_hardware(payload)
    payload["evidence_identity"] = compute_evidence_identity(payload)

    with pytest.raises(
        ValueError,
        match="EXECUTION_CONTRACT_CANONICAL_RUNTIME_VERSION_MISMATCH",
    ):
        authoritative_validate_probe_artifact(payload)


def test_runtime_fingerprint_rejects_arbitrary_nonempty_version() -> None:
    runtime = _runtime("arbitrary-nonempty-runtime-version")
    with pytest.raises(ValueError, match="PROBE_RUNTIME_VERSION_MISMATCH"):
        validate_runtime_fingerprint(runtime)


def test_runtime_1_0_artifact_remains_valid() -> None:
    payload = _probe()
    assert payload["runtime"]["profile_version"] == CANONICAL_RUNTIME_VERSION
    assert (
        payload["execution_contract"]["canonical_runtime_version"]
        == CANONICAL_RUNTIME_VERSION
    )
    assert validate_runtime_fingerprint(payload["runtime"]) is payload["runtime"]
    assert (
        validate_execution_contract(payload["execution_contract"])
        is payload["execution_contract"]
    )
    assert authoritative_validate_probe_artifact(payload) is payload
