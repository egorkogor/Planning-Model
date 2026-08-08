from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.canonical_probe_evidence_validation import validate_runtime_cross_binding
from scripts.canonical_training_probe_contract import (
    EXECUTION_CONTRACT_VERSION,
    PROBE_VERSION,
    _resolve_profile_before_torch_import,
    validate_execution_contract,
)
from scripts.canonical_training_probe_parity import (
    _capture_parameter_order,
    _first_parity_difference,
    _parameter_names_for_objects,
)
from scripts.run_canonical_training_probe import (
    compare_probes,
    compute_evidence_identity,
    compute_probe_identity,
    validate_probe_artifact,
    validate_probe_identity,
)

TASK_IDS = ["bw-00000001", "bw-00000002", "bw-00000003"]
FINGERPRINT_VERSION = "toy-quality-cpu-hardware-fingerprint/1.0"
SOFTWARE_FIELDS = (
    "canonical_runtime_version",
    "python_implementation",
    "python_version",
    "python_compiler",
    "python_build",
    "torch_version",
    "torch_build_configuration_sha256",
    "mkl_available",
    "openmp_available",
    "mkldnn_available",
)
CONTRACT_ONLY_CROSS_BINDING_FIELDS = (
    "python_version",
    "python_compiler",
    "torch_version",
    "torch_build_configuration_sha256",
    "actual_atten_cpu_capability",
    "mkl_available",
    "openmp_available",
    "mkldnn_available",
)
_BUILD_CONFIGURATION = "fixture torch build configuration"
_BUILD_CONFIGURATION_SHA256 = "sha256:" + hashlib.sha256(
    _BUILD_CONFIGURATION.encode("utf-8")
).hexdigest()


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _contract(profile: str = "historical-default") -> dict:
    controlled = profile != "historical-default"
    return {
        "contract_version": EXECUTION_CONTRACT_VERSION,
        "profile": profile,
        "profile_kind": "controlled-investigation" if controlled else "historical",
        "ATEN_CPU_CAPABILITY": (
            "default"
            if profile == "default-single-tensor"
            else "avx2"
            if controlled
            else None
        ),
        "actual_atten_cpu_capability": (
            "DEFAULT" if profile == "default-single-tensor" else "AVX2"
        ),
        "MKL_CBWR": "COMPATIBLE" if controlled else None,
        "foreach": False if controlled else None,
        "fused": False if controlled else None,
        "optimizer_class": "torch.optim.adamw.AdamW",
        "optimizer_hyperparameters": {
            "amsgrad": False,
            "betas": [0.9, 0.95],
            "capturable": False,
            "decoupled_weight_decay": True,
            "differentiable": False,
            "eps": 1e-8,
            "lr": 0.0003,
            "maximize": False,
            "weight_decay": 0.01,
        },
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "mkldnn_enabled": False,
        "deterministic_algorithms": {"enabled": True, "warn_only": False},
        "canonical_runtime_version": "toy-quality-canonical-cpu-runtime/1.0",
        "python_implementation": "CPython",
        "python_version": "3.11.15",
        "python_compiler": "GCC 13.3.0",
        "python_build": ["main", "2026-07-01"],
        "torch_version": "2.12.0+cpu",
        "torch_build_configuration_sha256": _BUILD_CONFIGURATION_SHA256,
        "mkl_available": True,
        "openmp_available": True,
        "mkldnn_available": True,
    }


def _runtime(version: str = "toy-quality-canonical-cpu-runtime/1.0") -> dict:
    return {
        "profile_version": version,
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


def _hardware(runtime: dict, contract: dict) -> dict:
    observation = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "os": {
            "system": "Linux",
            "release": "6.8.0",
            "version": "fixture",
            "machine": "x86_64",
            "architecture": "64bit",
            "os_release": {"ID": "ubuntu", "VERSION_ID": "24.04"},
        },
        "cpu": {
            "vendor": "GenuineIntel",
            "family": "6",
            "model": "85",
            "stepping": "7",
            "model_name": "Fixture CPU",
            "microcode": "0x1",
            "logical_cpu_count": 2,
            "flags_sha256": "sha256:" + "1" * 64,
            "capabilities": {
                "sse2": True,
                "avx": True,
                "avx2": True,
                "avx512f": False,
                "avx512dq": False,
                "avx512bw": False,
                "avx512vl": False,
                "fma": True,
            },
        },
        "runner": {
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
            "RUNNER_ENVIRONMENT": "github-hosted",
            "ImageOS": "ubuntu24",
            "ImageVersion": "20260720.247.2",
            "AZURE_REGION": "fixture-region",
        },
        "python": {
            "implementation": contract["python_implementation"],
            "version": contract["python_version"],
            "compiler": contract["python_compiler"],
            "build_number": contract["python_build"][0],
            "build_date": contract["python_build"][1],
        },
        "pytorch": {
            "version": contract["torch_version"],
            "cuda_version": None,
            "build_configuration": _BUILD_CONFIGURATION,
            "build_configuration_sha256": contract[
                "torch_build_configuration_sha256"
            ],
            "cpu_dispatch_capability": contract["actual_atten_cpu_capability"],
            "mkl_available": contract["mkl_available"],
            "openmp_available": contract["openmp_available"],
            "mkldnn_available": contract["mkldnn_available"],
            "mkldnn_enabled": contract["mkldnn_enabled"],
        },
        "canonical_runtime": copy.deepcopy(runtime),
        "execution_environment": {
            "OMP_NUM_THREADS": runtime["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": runtime["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": runtime["OPENBLAS_NUM_THREADS"],
            "NUMEXPR_NUM_THREADS": runtime["NUMEXPR_NUM_THREADS"],
            "ATEN_CPU_CAPABILITY": contract["ATEN_CPU_CAPABILITY"],
            "MKL_CBWR": contract["MKL_CBWR"],
        },
    }
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "observation_identity_sha256": _hash(observation),
        "observed_runtime_and_hardware": observation,
    }


def _update(index: int, parameter_names: list[str]) -> dict:
    mapping = {name: _hash([index, name]) for name in parameter_names}
    return {
        "update": index,
        "epoch": ((index - 1) // 3) + 1,
        "task_id": TASK_IDS[(index - 1) % 3],
        "encoded_task_sha256": _hash(["encoded", index]),
        "forward_logits": {
            "action": _hash(["action", index]),
            "arg1": _hash(["arg1", index]),
            "arg2": _hash(["arg2", index]),
            "z_semantic": None,
        },
        "loss_components": {
            "action": _hash(["loss-action", index]),
            "arg1": _hash(["loss-arg1", index]),
            "arg2": _hash(["loss-arg2", index]),
            "total": _hash(["loss-total", index]),
        },
        "raw_gradients": copy.deepcopy(mapping),
        "gradient_norm": _hash(["norm", index]),
        "gradients_after_clipping": copy.deepcopy(mapping),
        "adamw_exp_avg": copy.deepcopy(mapping),
        "adamw_exp_avg_sq": copy.deepcopy(mapping),
        "parameters_after_optimizer_step": copy.deepcopy(mapping),
    }


def _probe(contract: dict | None = None) -> dict:
    selected = copy.deepcopy(contract or _contract())
    runtime = _runtime(selected["canonical_runtime_version"])
    parameter_names = ["a", "b"]
    payload = {
        "probe_version": PROBE_VERSION,
        "variant": "A2",
        "seed": 17,
        "epochs": 3,
        "ordered_train_task_ids": list(TASK_IDS),
        "parameter_names": parameter_names,
        "initial_parameters": {
            name: _hash(["initial", name]) for name in parameter_names
        },
        "updates": [_update(index, parameter_names) for index in range(1, 10)],
        "execution_contract": selected,
        "execution_contract_sha256": _hash(selected),
        "runtime": runtime,
        "hardware_runtime_fingerprint": _hardware(runtime, selected),
    }
    payload["probe_identity"] = compute_probe_identity(payload)
    payload["evidence_identity"] = compute_evidence_identity(payload)
    validate_probe_artifact(payload)
    return payload


def _reseal(payload: dict) -> None:
    payload["execution_contract_sha256"] = _hash(payload["execution_contract"])
    payload["probe_identity"] = compute_probe_identity(payload)
    payload["evidence_identity"] = compute_evidence_identity(payload)


def _reseal_hardware(payload: dict) -> None:
    fingerprint = payload["hardware_runtime_fingerprint"]
    observation = fingerprint["observed_runtime_and_hardware"]
    fingerprint["observation_identity_sha256"] = _hash(observation)


def _reseal_hardware_and_evidence(payload: dict) -> None:
    _reseal_hardware(payload)
    payload["evidence_identity"] = compute_evidence_identity(payload)


@pytest.mark.parametrize("field", sorted(_contract()))
def test_contract_rejects_missing_field_even_after_hash_recompute(field: str) -> None:
    contract = _contract()
    del contract[field]
    with pytest.raises(ValueError, match="EXECUTION_CONTRACT_FIELDS_MISMATCH"):
        validate_execution_contract(contract)


def test_contract_rejects_extra_field_even_after_hash_recompute() -> None:
    contract = _contract()
    contract["accepted_target"] = True
    with pytest.raises(ValueError, match="EXECUTION_CONTRACT_FIELDS_MISMATCH"):
        validate_execution_contract(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    (("foreach", True), ("fused", True), ("MKL_CBWR", None)),
)
def test_controlled_profile_rejects_incoherent_field(field: str, value: object) -> None:
    contract = _contract("default-single-tensor")
    contract[field] = value
    with pytest.raises(ValueError, match="EXECUTION_CONTRACT_PROFILE_FIELD_MISMATCH"):
        validate_execution_contract(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("torch_num_threads", 2),
        ("torch_num_interop_threads", 2),
        ("mkldnn_enabled", True),
    ),
)
def test_contract_rejects_noncanonical_runtime_value(field: str, value: object) -> None:
    contract = _contract()
    contract[field] = value
    with pytest.raises(ValueError, match="EXECUTION_CONTRACT_PROFILE_FIELD_MISMATCH"):
        validate_execution_contract(contract)


def test_contract_rejects_deterministic_algorithm_drift() -> None:
    contract = _contract()
    contract["deterministic_algorithms"]["enabled"] = False
    with pytest.raises(ValueError, match="EXECUTION_CONTRACT_DETERMINISTIC_VALUE_MISMATCH"):
        validate_execution_contract(contract)


def test_contract_rejects_optimizer_hyperparameter_mutation() -> None:
    contract = _contract()
    contract["optimizer_hyperparameters"]["lr"] = 0.1
    with pytest.raises(ValueError, match="OPTIMIZER_HYPERPARAMETERS_MISMATCH"):
        validate_execution_contract(contract)


@pytest.mark.parametrize(
    ("profile", "foreach", "fused"),
    (
        ("historical-default", False, None),
        ("default-single-tensor", True, False),
        ("avx2-single-tensor", False, True),
    ),
)
def test_named_profile_flags_are_confirmations_not_overrides(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    foreach: bool | None,
    fused: bool | None,
) -> None:
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    monkeypatch.delenv("MKL_CBWR", raising=False)
    if profile == "default-single-tensor":
        monkeypatch.setenv("ATEN_CPU_CAPABILITY", "default")
        monkeypatch.setenv("MKL_CBWR", "COMPATIBLE")
    elif profile == "avx2-single-tensor":
        monkeypatch.setenv("ATEN_CPU_CAPABILITY", "avx2")
        monkeypatch.setenv("MKL_CBWR", "COMPATIBLE")
    with pytest.raises(RuntimeError, match="EXECUTION_PROFILE_OPTIMIZER_FLAG_MISMATCH"):
        _resolve_profile_before_torch_import(
            profile,
            optimizer_foreach=foreach,
            optimizer_fused=fused,
        )


def test_probe_rejects_stale_identity_after_numerical_mutation() -> None:
    payload = _probe()
    payload["updates"][0]["raw_gradients"]["a"] = _hash("mutated")
    with pytest.raises(ValueError, match="PROBE_IDENTITY_HASH_MISMATCH"):
        validate_probe_artifact(payload)


def test_probe_rejects_new_probe_identity_with_stale_evidence_identity() -> None:
    payload = _probe()
    payload["updates"][0]["raw_gradients"]["a"] = _hash("mutated")
    payload["probe_identity"] = compute_probe_identity(payload)
    with pytest.raises(ValueError, match="EVIDENCE_IDENTITY_HASH_MISMATCH"):
        validate_probe_artifact(payload)


def test_probe_rejects_hardware_mutation_with_stale_fingerprint_hash() -> None:
    payload = _probe()
    observation = payload["hardware_runtime_fingerprint"][
        "observed_runtime_and_hardware"
    ]
    observation["cpu"]["model_name"] = "mutated"
    with pytest.raises(ValueError, match="HARDWARE_FINGERPRINT_OBSERVATION_HASH_MISMATCH"):
        validate_probe_artifact(payload)


def test_probe_rejects_resealed_hardware_with_stale_evidence_identity() -> None:
    payload = _probe()
    observation = payload["hardware_runtime_fingerprint"][
        "observed_runtime_and_hardware"
    ]
    observation["cpu"]["model_name"] = "mutated"
    _reseal_hardware(payload)
    with pytest.raises(ValueError, match="EVIDENCE_IDENTITY_HASH_MISMATCH"):
        validate_probe_artifact(payload)


def test_probe_rejects_invalid_contract_with_recomputed_contract_hash() -> None:
    payload = _probe()
    payload["execution_contract"]["torch_num_threads"] = 2
    payload["execution_contract_sha256"] = _hash(payload["execution_contract"])
    with pytest.raises(ValueError, match="EXECUTION_CONTRACT_PROFILE_FIELD_MISMATCH"):
        validate_probe_artifact(payload)


def test_invalid_artifact_is_rejected_before_contract_mismatch_result() -> None:
    left = _probe(_contract("avx2-single-tensor"))
    right = _probe(_contract("default-single-tensor"))
    left["execution_contract"]["torch_num_threads"] = 2
    left["execution_contract_sha256"] = _hash(left["execution_contract"])
    with pytest.raises(ValueError, match="EXECUTION_CONTRACT_PROFILE_FIELD_MISMATCH"):
        compare_probes(left, right)


def test_probe_rejects_extra_top_level_field() -> None:
    payload = _probe()
    payload["accepted_target"] = True
    with pytest.raises(ValueError, match="PROBE_ARTIFACT_FIELDS_MISMATCH"):
        validate_probe_artifact(payload)


def test_probe_rejects_missing_runtime() -> None:
    payload = _probe()
    del payload["runtime"]
    with pytest.raises(ValueError, match="PROBE_ARTIFACT_FIELDS_MISMATCH"):
        validate_probe_artifact(payload)


def test_probe_rejects_stale_evidence_identity() -> None:
    payload = _probe()
    payload["evidence_identity"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="EVIDENCE_IDENTITY_HASH_MISMATCH"):
        validate_probe_artifact(payload)


def test_same_contract_and_exact_numbers_compare_equal() -> None:
    left = _probe()
    right = copy.deepcopy(left)
    result = compare_probes(left, right)
    assert result["comparable"] is True
    assert result["equal"] is True


def test_same_contract_and_resealed_mutation_compare_different() -> None:
    left = _probe()
    right = copy.deepcopy(left)
    right["updates"][0]["adamw_exp_avg_sq"]["a"] = _hash("mutated")
    right["probe_identity"] = compute_probe_identity(right)
    right["evidence_identity"] = compute_evidence_identity(right)
    result = compare_probes(left, right)
    assert result["comparable"] is True
    assert result["equal"] is False
    assert result["first_divergence"]["stage"] == "adamw_exp_avg_sq"


@pytest.mark.parametrize("field", CONTRACT_ONLY_CROSS_BINDING_FIELDS)
def test_resealed_contract_only_software_mutation_is_rejected(field: str) -> None:
    payload = _probe()
    values = {
        "python_version": "3.13.7",
        "python_compiler": "Clang 20.0",
        "torch_version": "2.12.1+cpu",
        "torch_build_configuration_sha256": "sha256:" + "b" * 64,
        "actual_atten_cpu_capability": "OTHER",
        "mkl_available": False,
        "openmp_available": False,
        "mkldnn_available": False,
    }
    payload["execution_contract"][field] = values[field]
    _reseal(payload)
    with pytest.raises(
        ValueError, match=f"PROBE_RUNTIME_CROSS_BINDING_MISMATCH:{field}"
    ):
        validate_probe_artifact(payload)


@pytest.mark.parametrize(
    ("path", "value", "error_field"),
    (
        (("python", "version"), "3.13.7", "python_version"),
        (("pytorch", "version"), "2.12.1+cpu", "torch_version"),
        (("pytorch", "cpu_dispatch_capability"), "OTHER", "actual_atten_cpu_capability"),
        (("execution_environment", "MKL_CBWR"), "COMPATIBLE", "MKL_CBWR"),
        (
            ("canonical_runtime", "profile_version"),
            "toy-quality-canonical-cpu-runtime/1.1",
            "canonical_runtime_version",
        ),
    ),
)
def test_resealed_hardware_only_cross_binding_mutation_is_rejected(
    path: tuple[str, str], value: object, error_field: str
) -> None:
    payload = _probe()
    observation = payload["hardware_runtime_fingerprint"][
        "observed_runtime_and_hardware"
    ]
    section, field = path
    observation[section][field] = value
    _reseal_hardware_and_evidence(payload)
    with pytest.raises(
        ValueError, match=f"PROBE_RUNTIME_CROSS_BINDING_MISMATCH:{error_field}"
    ):
        validate_probe_artifact(payload)


def test_full_runtime_must_equal_hardware_canonical_runtime() -> None:
    payload = _probe()
    hardware_runtime = payload["hardware_runtime_fingerprint"][
        "observed_runtime_and_hardware"
    ]["canonical_runtime"]
    hardware_runtime["profile_version"] = payload["runtime"]["profile_version"]
    hardware_runtime = copy.deepcopy(payload["runtime"])
    hardware_runtime["deterministic_algorithms_enabled"] = False
    payload["hardware_runtime_fingerprint"]["observed_runtime_and_hardware"][
        "canonical_runtime"
    ] = hardware_runtime
    _reseal_hardware_and_evidence(payload)
    with pytest.raises(ValueError, match="PROBE_RUNTIME_VALUE_MISMATCH"):
        validate_probe_artifact(payload)


def test_software_runtime_contract_mutation_makes_valid_artifacts_incomparable() -> None:
    left = _probe()
    right = copy.deepcopy(left)
    right["execution_contract"]["python_version"] = "3.13.7"
    observation = right["hardware_runtime_fingerprint"][
        "observed_runtime_and_hardware"
    ]
    observation["python"]["version"] = "3.13.7"
    _reseal_hardware(right)
    _reseal(right)
    validate_probe_artifact(left)
    validate_probe_artifact(right)
    report = compare_probes(left, right)
    assert report["comparable"] is False
    assert report["reason"] == "EXECUTION_CONTRACT_MISMATCH"
    assert report["equal"] is None


def test_runtime_cross_binding_accepts_matching_historical_dispatch() -> None:
    payload = _probe()
    validate_runtime_cross_binding(
        payload["execution_contract"],
        payload["runtime"],
        payload["hardware_runtime_fingerprint"],
    )


def test_probe_2_0_requires_exact_epochs() -> None:
    payload = _probe()
    payload["epochs"] = 1
    with pytest.raises(ValueError, match="PROBE_EPOCHS_MISMATCH"):
        validate_probe_identity(payload)


def test_probe_2_0_requires_exact_task_ids() -> None:
    payload = _probe()
    payload["ordered_train_task_ids"] = list(reversed(TASK_IDS))
    with pytest.raises(ValueError, match="PROBE_TASK_IDS_MISMATCH"):
        validate_probe_identity(payload)


def test_probe_2_0_requires_nine_updates() -> None:
    payload = _probe()
    payload["updates"].pop()
    with pytest.raises(ValueError, match="PROBE_UPDATE_COUNT_MISMATCH"):
        validate_probe_identity(payload)


def test_probe_2_0_rejects_noncanonical_scalar_hash() -> None:
    payload = _probe()
    payload["updates"][0]["gradient_norm"] = "not-a-hash"
    with pytest.raises(ValueError, match="PROBE_HASH_FORMAT_INVALID:gradient_norm"):
        validate_probe_identity(payload)


def test_probe_2_0_rejects_gradient_key_outside_parameter_names() -> None:
    payload = _probe()
    payload["updates"][0]["raw_gradients"]["outside"] = _hash("x")
    with pytest.raises(ValueError, match="PROBE_HASH_MAPPING_PARAMETER_MISMATCH"):
        validate_probe_identity(payload)


def test_probe_2_0_requires_exact_forward_keys() -> None:
    payload = _probe()
    payload["updates"][0]["forward_logits"]["extra"] = _hash("x")
    with pytest.raises(ValueError, match="PROBE_FORWARD_LOGITS_FIELDS_MISMATCH"):
        validate_probe_identity(payload)


def test_probe_2_0_requires_exact_loss_keys() -> None:
    payload = _probe()
    del payload["updates"][0]["loss_components"]["arg2"]
    with pytest.raises(ValueError, match="PROBE_LOSS_COMPONENT_FIELDS_MISMATCH"):
        validate_probe_identity(payload)


def test_probe_2_0_requires_full_parameter_snapshots() -> None:
    payload = _probe()
    del payload["updates"][0]["parameters_after_optimizer_step"]["b"]
    with pytest.raises(ValueError, match="PROBE_HASH_MAPPING_PARAMETER_MISMATCH"):
        validate_probe_identity(payload)


def test_probe_2_0_requires_unique_parameter_names() -> None:
    payload = _probe()
    payload["parameter_names"] = ["a", "a"]
    with pytest.raises(ValueError, match="PROBE_PARAMETER_NAMES_DUPLICATE"):
        validate_probe_identity(payload)


class _FakeModel:
    def __init__(self, values: list[tuple[str, object]]) -> None:
        self._values = values

    def named_parameters(self) -> list[tuple[str, object]]:
        return list(self._values)


def test_parameter_capture_helper_before_adamw_is_order_independent() -> None:
    a = object()
    b = object()
    model = _FakeModel([("a", a), ("b", b)])
    captured: dict = {}
    helper_values = [("a", a), ("b", b)]
    _capture_parameter_order(captured, model=model, named_parameters=helper_values)
    adam_names = _parameter_names_for_objects(model, [a, b])
    adam_values = list(zip(adam_names, [a, b], strict=True))
    _capture_parameter_order(captured, model=model, named_parameters=adam_values)
    assert captured["parameter_names"] == ["a", "b"]
    assert captured["named_parameters"] == helper_values


def test_parameter_capture_adamw_before_helper_is_order_independent() -> None:
    a = object()
    b = object()
    model = _FakeModel([("a", a), ("b", b)])
    captured: dict = {}
    adam_names = _parameter_names_for_objects(model, [a, b])
    adam_values = list(zip(adam_names, [a, b], strict=True))
    _capture_parameter_order(captured, model=model, named_parameters=adam_values)
    helper_values = [("a", a), ("b", b)]
    _capture_parameter_order(captured, model=model, named_parameters=helper_values)
    assert captured["parameter_names"] == ["a", "b"]
    assert captured["named_parameters"] == helper_values


@pytest.mark.parametrize(
    "field",
    (
        "parameter_names",
        "optimizer_defaults",
        "initial_parameters",
        "encoded_task_sha256",
        "forward_logits",
        "loss_components",
        "raw_gradients",
        "gradient_norm",
        "gradients_after_clipping",
        "adamw_exp_avg",
        "adamw_exp_avg_sq",
        "parameters_after_optimizer_step",
        "gradient_clip_max_norm",
        "gradient_clip_parameter_names",
        "update_events",
    ),
)
def test_parity_comparison_detects_semantic_mutation(field: str) -> None:
    baseline = {
        "parameter_names": ["a"],
        "optimizer_defaults": {"lr": 0.0003},
        "initial_parameters": {"a": "hash"},
        "encoded_task_sha256": "hash",
        "forward_logits": {"action": "hash"},
        "loss_components": {"total": "hash"},
        "raw_gradients": {"a": "hash"},
        "gradient_norm": "hash",
        "gradients_after_clipping": {"a": "hash"},
        "adamw_exp_avg": {"a": "hash"},
        "adamw_exp_avg_sq": {"a": "hash"},
        "parameters_after_optimizer_step": {"a": "hash"},
        "gradient_clip_max_norm": 1.0,
        "gradient_clip_parameter_names": ["a"],
        "update_events": ["zero_grad", "forward", "loss", "backward", "clip", "step"],
    }
    mutated = copy.deepcopy(baseline)
    mutated[field] = "mutated"
    assert _first_parity_difference(baseline, mutated) == field
