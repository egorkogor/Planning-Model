from __future__ import annotations

import copy
import importlib

import pytest

from scripts.canonical_probe_evidence_validation import (
    validate_probe_artifact as authoritative_validate_probe_artifact,
)
from scripts.canonical_training_probe_contract import (
    compute_evidence_identity,
    compute_probe_identity,
)
from scripts.run_canonical_training_probe import (
    validate_probe_artifact as cli_validate_probe_artifact,
)
from tests.toy.test_canonical_training_probe_hardening import (
    _hash,
    _probe,
    _reseal_hardware,
)

probe_contract = importlib.import_module("scripts.canonical_training_probe_contract")
probe_core = importlib.import_module("scripts.canonical_training_probe_core")


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
