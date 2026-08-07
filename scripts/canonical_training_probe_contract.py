"""Execution-contract and retained probe-artifact validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from typing import Any

PROBE_VERSION = "toy-quality-canonical-training-probe/2.0"
EXECUTION_CONTRACT_VERSION = "toy-quality-cpu-execution-contract/1.0"
COMPARISON_VERSION = "toy-quality-canonical-training-probe-comparison/2.0"
PARITY_VERSION = "toy-quality-canonical-training-parity/2.0"
CANONICAL_RUNTIME_VERSION = "toy-quality-canonical-cpu-runtime/1.0"
EPOCHS = 3
_ORDERED_TRAIN_TASK_IDS = ["bw-00000001", "bw-00000002", "bw-00000003"]
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")

_EXECUTION_CONTRACT_FIELDS = {
    "contract_version",
    "profile",
    "profile_kind",
    "ATEN_CPU_CAPABILITY",
    "actual_atten_cpu_capability",
    "MKL_CBWR",
    "foreach",
    "fused",
    "optimizer_class",
    "optimizer_hyperparameters",
    "torch_num_threads",
    "torch_num_interop_threads",
    "mkldnn_enabled",
    "deterministic_algorithms",
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
}
_DETERMINISTIC_ALGORITHM_FIELDS = {"enabled", "warn_only"}
_OPTIMIZER_HYPERPARAMETERS = {
    "amsgrad": False,
    "betas": [0.9, 0.95],
    "capturable": False,
    "decoupled_weight_decay": True,
    "differentiable": False,
    "eps": 1e-8,
    "lr": 3e-4,
    "maximize": False,
    "weight_decay": 0.01,
}
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
_PROBE_ARTIFACT_FIELDS = {
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
    "runtime",
    "hardware_runtime_fingerprint",
    "probe_identity",
    "evidence_identity",
}
_UPDATE_FIELDS = {
    "update",
    "epoch",
    "task_id",
    "encoded_task_sha256",
    "forward_logits",
    "loss_components",
    "raw_gradients",
    "gradient_norm",
    "gradients_after_clipping",
    "adamw_exp_avg",
    "adamw_exp_avg_sq",
    "parameters_after_optimizer_step",
}
_FORWARD_LOGIT_FIELDS = {"action", "arg1", "arg2", "z_semantic"}
_LOSS_COMPONENT_FIELDS = {"action", "arg1", "arg2", "total"}
_RUNTIME_FIELDS = {
    "profile_version",
    "deterministic_algorithms_enabled",
    "deterministic_warn_only_enabled",
    "torch_num_threads",
    "torch_num_interop_threads",
    "mkldnn_enabled",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
}
_PROFILE_SPECS: dict[str, dict[str, object]] = {
    "historical-default": {
        "kind": "historical",
        "ATEN_CPU_CAPABILITY": None,
        "expected_actual_atten_cpu_capability": None,
        "MKL_CBWR": None,
        "foreach": None,
        "fused": None,
    },
    "avx2-single-tensor": {
        "kind": "controlled-investigation",
        "ATEN_CPU_CAPABILITY": "avx2",
        "expected_actual_atten_cpu_capability": "AVX2",
        "MKL_CBWR": "COMPATIBLE",
        "foreach": False,
        "fused": False,
    },
    "default-single-tensor": {
        "kind": "controlled-investigation",
        "ATEN_CPU_CAPABILITY": "default",
        "expected_actual_atten_cpu_capability": "DEFAULT",
        "MKL_CBWR": "COMPATIBLE",
        "foreach": False,
        "fused": False,
    },
}


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


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _ensure_torch_not_imported() -> None:
    if "torch" in sys.modules:
        raise RuntimeError("EXECUTION_PROFILE_TORCH_ALREADY_IMPORTED")


def _resolve_profile_before_torch_import(
    profile_name: str,
    *,
    optimizer_foreach: bool | None = None,
    optimizer_fused: bool | None = None,
) -> dict[str, object]:
    if profile_name not in _PROFILE_SPECS:
        raise ValueError(f"UNKNOWN_EXECUTION_PROFILE:{profile_name}")
    spec = dict(_PROFILE_SPECS[profile_name])
    for name in ("ATEN_CPU_CAPABILITY", "MKL_CBWR"):
        expected = spec[name]
        actual = os.environ.get(name)
        if expected is None and actual is not None:
            raise RuntimeError(
                f"EXECUTION_PROFILE_ENV_UNEXPECTED:{name}:actual={actual}"
            )
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"EXECUTION_PROFILE_ENV_MISMATCH:{name}:"
                f"expected={expected}:actual={actual}"
            )
    for name, requested in (
        ("foreach", optimizer_foreach),
        ("fused", optimizer_fused),
    ):
        if requested is not None and requested != spec[name]:
            raise RuntimeError(
                "EXECUTION_PROFILE_OPTIMIZER_FLAG_MISMATCH:"
                f"{name}:expected={spec[name]}:actual={requested}"
            )
    if spec["kind"] == "controlled-investigation" and (
        spec["foreach"] is None or spec["fused"] is None
    ):
        raise RuntimeError("CONTROLLED_PROFILE_OPTIMIZER_FLAG_UNSET")
    spec["profile"] = profile_name
    return spec


def _load_modules(seed: int) -> tuple[dict[str, object], dict[str, Any]]:
    import numpy as np
    import torch
    import torch.nn.functional as functional

    from planner_toy.canonical_runtime import configure_canonical_cpu_runtime
    from planner_toy.dataset import generate
    from planner_toy.model import LockedPlanner, canonical_task_encoding
    from planner_toy.semantic import targets
    from planner_toy.training import ACTIONS, labels
    from scripts.canonical_cpu_hardware_fingerprint import (
        full_hardware_runtime_fingerprint,
        validate_hardware_runtime_fingerprint,
    )

    runtime = configure_canonical_cpu_runtime(seed)
    np.random.seed(seed)
    return runtime, {
        "torch": torch,
        "functional": functional,
        "configure": configure_canonical_cpu_runtime,
        "generate": generate,
        "LockedPlanner": LockedPlanner,
        "canonical_task_encoding": canonical_task_encoding,
        "targets": targets,
        "actions": ACTIONS,
        "labels": labels,
        "full_hardware_runtime_fingerprint": full_hardware_runtime_fingerprint,
        "validate_hardware_runtime_fingerprint": (
            validate_hardware_runtime_fingerprint
        ),
    }


def _normalize_value(value: object) -> object:
    if isinstance(value, tuple | list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(item)
            for key, item in sorted(value.items())
        }
    return value


def _optimizer_hyperparameters(optimizer: Any) -> dict[str, object]:
    return {
        key: _normalize_value(value)
        for key, value in sorted(optimizer.defaults.items())
        if key not in {"foreach", "fused", "params"}
    }


def _build_optimizer(
    torch_module: Any,
    parameters: list[Any],
    spec: dict[str, object],
) -> Any:
    return torch_module.optim.AdamW(
        parameters,
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        foreach=spec["foreach"],
        fused=spec["fused"],
    )


def _torch_build_configuration_sha256(torch_module: Any) -> str:
    configuration = "\n".join(
        line.rstrip()
        for line in torch_module.__config__.show().strip().splitlines()
    )
    return _sha256_bytes(configuration.encode("utf-8"))


def validate_execution_contract(contract: object) -> dict[str, object]:
    if not isinstance(contract, dict):
        raise ValueError("EXECUTION_CONTRACT_NOT_OBJECT")
    if set(contract) != _EXECUTION_CONTRACT_FIELDS:
        raise ValueError("EXECUTION_CONTRACT_FIELDS_MISMATCH")
    if contract["contract_version"] != EXECUTION_CONTRACT_VERSION:
        raise ValueError("EXECUTION_CONTRACT_VERSION_MISMATCH")
    profile = contract["profile"]
    if not isinstance(profile, str) or profile not in _PROFILE_SPECS:
        raise ValueError("EXECUTION_CONTRACT_PROFILE_INVALID")
    spec = _PROFILE_SPECS[profile]
    expected_values = {
        "profile_kind": spec["kind"],
        "ATEN_CPU_CAPABILITY": spec["ATEN_CPU_CAPABILITY"],
        "MKL_CBWR": spec["MKL_CBWR"],
        "foreach": spec["foreach"],
        "fused": spec["fused"],
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "mkldnn_enabled": False,
    }
    for name, expected in expected_values.items():
        if contract[name] != expected:
            raise ValueError(f"EXECUTION_CONTRACT_PROFILE_FIELD_MISMATCH:{name}")
    actual_dispatch = contract["actual_atten_cpu_capability"]
    if not isinstance(actual_dispatch, str) or not actual_dispatch:
        raise ValueError("EXECUTION_CONTRACT_ATEN_DISPATCH_INVALID")
    expected_dispatch = spec["expected_actual_atten_cpu_capability"]
    if expected_dispatch is not None and actual_dispatch != expected_dispatch:
        raise ValueError("EXECUTION_CONTRACT_ATEN_DISPATCH_MISMATCH")
    if contract["optimizer_class"] != "torch.optim.adamw.AdamW":
        raise ValueError("EXECUTION_CONTRACT_OPTIMIZER_CLASS_MISMATCH")
    if contract["optimizer_hyperparameters"] != _OPTIMIZER_HYPERPARAMETERS:
        raise ValueError("EXECUTION_CONTRACT_OPTIMIZER_HYPERPARAMETERS_MISMATCH")
    deterministic = contract["deterministic_algorithms"]
    if not isinstance(deterministic, dict) or set(deterministic) != (
        _DETERMINISTIC_ALGORITHM_FIELDS
    ):
        raise ValueError("EXECUTION_CONTRACT_DETERMINISTIC_FIELDS_MISMATCH")
    if deterministic != {"enabled": True, "warn_only": False}:
        raise ValueError("EXECUTION_CONTRACT_DETERMINISTIC_VALUE_MISMATCH")
    _require_nonempty_string(
        contract["canonical_runtime_version"],
        "EXECUTION_CONTRACT_SOFTWARE_FIELD_INVALID:canonical_runtime_version",
    )
    for name in (
        "python_implementation",
        "python_version",
        "python_compiler",
        "torch_version",
    ):
        _require_nonempty_string(
            contract[name], f"EXECUTION_CONTRACT_SOFTWARE_FIELD_INVALID:{name}"
        )
    python_build = contract["python_build"]
    if (
        not isinstance(python_build, list)
        or len(python_build) != 2
        or not all(isinstance(value, str) and value for value in python_build)
    ):
        raise ValueError("EXECUTION_CONTRACT_SOFTWARE_FIELD_INVALID:python_build")
    _require_hash(
        contract["torch_build_configuration_sha256"],
        "EXECUTION_CONTRACT_SOFTWARE_FIELD_INVALID:torch_build_configuration_sha256",
    )
    for name in ("mkl_available", "openmp_available", "mkldnn_available"):
        if type(contract[name]) is not bool:
            raise ValueError(f"EXECUTION_CONTRACT_SOFTWARE_FIELD_INVALID:{name}")
    return contract


def _execution_contract(
    *,
    spec: dict[str, object],
    optimizer: Any,
    torch_module: Any,
    runtime: dict[str, object],
) -> dict[str, object]:
    actual_dispatch = torch_module.backends.cpu.get_cpu_capability()
    expected_dispatch = spec["expected_actual_atten_cpu_capability"]
    if expected_dispatch is not None and actual_dispatch != expected_dispatch:
        raise RuntimeError(
            "EXECUTION_PROFILE_ATEN_DISPATCH_MISMATCH:"
            f"expected={expected_dispatch}:actual={actual_dispatch}"
        )
    python_build = list(platform.python_build())
    contract = {
        "contract_version": EXECUTION_CONTRACT_VERSION,
        "profile": spec["profile"],
        "profile_kind": spec["kind"],
        "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY"),
        "actual_atten_cpu_capability": actual_dispatch,
        "MKL_CBWR": os.environ.get("MKL_CBWR"),
        "foreach": optimizer.defaults.get("foreach"),
        "fused": optimizer.defaults.get("fused"),
        "optimizer_class": (
            f"{optimizer.__class__.__module__}."
            f"{optimizer.__class__.__qualname__}"
        ),
        "optimizer_hyperparameters": _optimizer_hyperparameters(optimizer),
        "torch_num_threads": torch_module.get_num_threads(),
        "torch_num_interop_threads": torch_module.get_num_interop_threads(),
        "mkldnn_enabled": torch_module.backends.mkldnn.enabled,
        "deterministic_algorithms": {
            "enabled": torch_module.are_deterministic_algorithms_enabled(),
            "warn_only": (
                torch_module.is_deterministic_algorithms_warn_only_enabled()
            ),
        },
        "canonical_runtime_version": runtime.get("profile_version"),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "python_build": python_build,
        "torch_version": torch_module.__version__,
        "torch_build_configuration_sha256": (
            _torch_build_configuration_sha256(torch_module)
        ),
        "mkl_available": torch_module.backends.mkl.is_available(),
        "openmp_available": torch_module.backends.openmp.is_available(),
        "mkldnn_available": torch_module.backends.mkldnn.is_available(),
    }
    validate_execution_contract(contract)
    return contract


def _validate_execution_contract_hash(
    payload: dict[str, object],
) -> dict[str, object]:
    contract = validate_execution_contract(payload.get("execution_contract"))
    identity = _require_hash(
        payload.get("execution_contract_sha256"),
        "EXECUTION_CONTRACT_HASH_FORMAT_INVALID",
    )
    if identity != _sha256_bytes(_canonical_bytes(contract)):
        raise ValueError("EXECUTION_CONTRACT_HASH_MISMATCH")
    return contract


def _probe_identity_payload(payload: dict[str, object]) -> dict[str, object]:
    try:
        return {key: payload[key] for key in _PROBE_IDENTITY_FIELDS}
    except KeyError as error:
        raise ValueError(f"PROBE_IDENTITY_FIELD_MISSING:{error.args[0]}") from None


def compute_probe_identity(payload: dict[str, object]) -> str:
    _validate_execution_contract_hash(payload)
    return _sha256_bytes(_canonical_bytes(_probe_identity_payload(payload)))


def _validate_hash_mapping(
    value: object,
    *,
    field: str,
    allowed_names: set[str],
    exact_names: set[str] | None = None,
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"PROBE_HASH_MAPPING_INVALID:{field}")
    names = set(value)
    if not names <= allowed_names:
        raise ValueError(f"PROBE_HASH_MAPPING_PARAMETER_MISMATCH:{field}")
    if exact_names is not None and names != exact_names:
        raise ValueError(f"PROBE_HASH_MAPPING_PARAMETER_MISMATCH:{field}")
    for item in value.values():
        _require_hash(item, f"PROBE_HASH_FORMAT_INVALID:{field}")
    return value


def _validate_update(
    update: object,
    *,
    index: int,
    parameter_names: list[str],
) -> None:
    if not isinstance(update, dict) or set(update) != _UPDATE_FIELDS:
        raise ValueError("PROBE_UPDATE_FIELDS_MISMATCH")
    expected_epoch = ((index - 1) // len(_ORDERED_TRAIN_TASK_IDS)) + 1
    expected_task = _ORDERED_TRAIN_TASK_IDS[
        (index - 1) % len(_ORDERED_TRAIN_TASK_IDS)
    ]
    if update["update"] != index:
        raise ValueError("PROBE_UPDATE_INDEX_MISMATCH")
    if update["epoch"] != expected_epoch:
        raise ValueError("PROBE_UPDATE_EPOCH_MISMATCH")
    if update["task_id"] != expected_task:
        raise ValueError("PROBE_UPDATE_TASK_ORDER_MISMATCH")
    _require_hash(update["encoded_task_sha256"], "PROBE_HASH_FORMAT_INVALID:encoded_task")
    _require_hash(update["gradient_norm"], "PROBE_HASH_FORMAT_INVALID:gradient_norm")
    forward = update["forward_logits"]
    if not isinstance(forward, dict) or set(forward) != _FORWARD_LOGIT_FIELDS:
        raise ValueError("PROBE_FORWARD_LOGITS_FIELDS_MISMATCH")
    for name in ("action", "arg1", "arg2"):
        _require_hash(forward[name], f"PROBE_HASH_FORMAT_INVALID:forward_logits.{name}")
    z_semantic = forward["z_semantic"]
    if z_semantic is not None:
        _require_hash(
            z_semantic,
            "PROBE_HASH_FORMAT_INVALID:forward_logits.z_semantic",
        )
    losses = update["loss_components"]
    if not isinstance(losses, dict) or set(losses) != _LOSS_COMPONENT_FIELDS:
        raise ValueError("PROBE_LOSS_COMPONENT_FIELDS_MISMATCH")
    for name, value in losses.items():
        _require_hash(value, f"PROBE_HASH_FORMAT_INVALID:loss_components.{name}")
    allowed = set(parameter_names)
    for field in (
        "raw_gradients",
        "gradients_after_clipping",
        "adamw_exp_avg",
        "adamw_exp_avg_sq",
    ):
        _validate_hash_mapping(update[field], field=field, allowed_names=allowed)
    _validate_hash_mapping(
        update["parameters_after_optimizer_step"],
        field="parameters_after_optimizer_step",
        allowed_names=allowed,
        exact_names=allowed,
    )


def validate_probe_identity(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("PROBE_NOT_OBJECT")
    if payload.get("probe_version") != PROBE_VERSION:
        raise ValueError("PROBE_VERSION_MISMATCH")
    if payload.get("variant") != "A2":
        raise ValueError("PROBE_VARIANT_MISMATCH")
    if type(payload.get("seed")) is not int:
        raise ValueError("PROBE_SEED_INVALID")
    if payload.get("epochs") != EPOCHS:
        raise ValueError("PROBE_EPOCHS_MISMATCH")
    if payload.get("ordered_train_task_ids") != _ORDERED_TRAIN_TASK_IDS:
        raise ValueError("PROBE_TASK_IDS_MISMATCH")
    parameter_names = payload.get("parameter_names")
    if not isinstance(parameter_names, list) or not parameter_names or not all(
        isinstance(value, str) and value for value in parameter_names
    ):
        raise ValueError("PROBE_PARAMETER_NAMES_INVALID")
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("PROBE_PARAMETER_NAMES_DUPLICATE")
    allowed = set(parameter_names)
    _validate_hash_mapping(
        payload.get("initial_parameters"),
        field="initial_parameters",
        allowed_names=allowed,
        exact_names=allowed,
    )
    updates = payload.get("updates")
    if not isinstance(updates, list) or len(updates) != 9:
        raise ValueError("PROBE_UPDATE_COUNT_MISMATCH")
    for index, update in enumerate(updates, start=1):
        _validate_update(update, index=index, parameter_names=parameter_names)
    _validate_execution_contract_hash(payload)
    identity = _require_hash(
        payload.get("probe_identity"),
        "PROBE_IDENTITY_HASH_FORMAT_INVALID",
    )
    if identity != compute_probe_identity(payload):
        raise ValueError("PROBE_IDENTITY_HASH_MISMATCH")
    return payload


def validate_runtime_fingerprint(runtime: object) -> dict[str, object]:
    if not isinstance(runtime, dict):
        raise ValueError("PROBE_RUNTIME_NOT_OBJECT")
    if set(runtime) != _RUNTIME_FIELDS:
        raise ValueError("PROBE_RUNTIME_FIELDS_MISMATCH")
    _require_nonempty_string(runtime["profile_version"], "PROBE_RUNTIME_VERSION_INVALID")
    expected_values = {
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
    for name, expected in expected_values.items():
        if runtime[name] != expected:
            raise ValueError(f"PROBE_RUNTIME_VALUE_MISMATCH:{name}")
    return runtime


def _evidence_identity_payload(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "evidence_identity"}


def compute_evidence_identity(payload: dict[str, object]) -> str:
    return _sha256_bytes(_canonical_bytes(_evidence_identity_payload(payload)))


def validate_probe_artifact(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("PROBE_NOT_OBJECT")
    if set(payload) != _PROBE_ARTIFACT_FIELDS:
        raise ValueError("PROBE_ARTIFACT_FIELDS_MISMATCH")
    contract = validate_execution_contract(payload["execution_contract"])
    _validate_execution_contract_hash(payload)
    validate_probe_identity(payload)
    from scripts.canonical_cpu_hardware_fingerprint import (
        validate_hardware_runtime_fingerprint,
    )

    validate_hardware_runtime_fingerprint(payload["hardware_runtime_fingerprint"])
    runtime = validate_runtime_fingerprint(payload["runtime"])
    if contract["canonical_runtime_version"] != runtime["profile_version"]:
        raise ValueError("PROBE_RUNTIME_CONTRACT_MISMATCH")
    evidence_identity = _require_hash(
        payload["evidence_identity"],
        "EVIDENCE_IDENTITY_HASH_FORMAT_INVALID",
    )
    if evidence_identity != compute_evidence_identity(payload):
        raise ValueError("EVIDENCE_IDENTITY_HASH_MISMATCH")
    return payload
