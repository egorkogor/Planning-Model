"""Exact update-level probe for canonical A2 CPU investigations.

Process-start controls are validated before PyTorch is imported. The retained
profiles reproduce the historical optimizer defaults or explicit single-tensor
investigation alternatives without modifying the frozen training path.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROBE_VERSION = "toy-quality-canonical-training-probe/2.0"
EXECUTION_CONTRACT_VERSION = "toy-quality-cpu-execution-contract/1.0"
COMPARISON_VERSION = "toy-quality-canonical-training-probe-comparison/2.0"
PARITY_VERSION = "toy-quality-canonical-training-parity/1.0"
EPOCHS = 3
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")

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
    if optimizer_foreach is not None:
        spec["foreach"] = optimizer_foreach
    if optimizer_fused is not None:
        spec["fused"] = optimizer_fused
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
        "full_hardware_runtime_fingerprint": (
            full_hardware_runtime_fingerprint
        ),
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


def _execution_contract(
    *,
    spec: dict[str, object],
    optimizer: Any,
    torch_module: Any,
) -> dict[str, object]:
    actual_dispatch = torch_module.backends.cpu.get_cpu_capability()
    expected_dispatch = spec["expected_actual_atten_cpu_capability"]
    if expected_dispatch is not None and actual_dispatch != expected_dispatch:
        raise RuntimeError(
            "EXECUTION_PROFILE_ATEN_DISPATCH_MISMATCH:"
            f"expected={expected_dispatch}:actual={actual_dispatch}"
        )
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
    }
    if contract["foreach"] != spec["foreach"]:
        raise RuntimeError("EXECUTION_PROFILE_FOREACH_MISMATCH")
    if contract["fused"] != spec["fused"]:
        raise RuntimeError("EXECUTION_PROFILE_FUSED_MISMATCH")
    return contract


def _validate_execution_contract_hash(
    payload: dict[str, object],
) -> dict[str, object]:
    contract = payload.get("execution_contract")
    identity = payload.get("execution_contract_sha256")
    if not isinstance(contract, dict):
        raise ValueError("EXECUTION_CONTRACT_MISSING")
    if (
        not isinstance(identity, str)
        or _HASH_PATTERN.fullmatch(identity) is None
    ):
        raise ValueError("EXECUTION_CONTRACT_HASH_FORMAT_INVALID")
    if identity != _sha256_bytes(_canonical_bytes(contract)):
        raise ValueError("EXECUTION_CONTRACT_HASH_MISMATCH")
    return contract


def _probe_identity_payload(payload: dict[str, object]) -> dict[str, object]:
    keys = (
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
    return {key: payload[key] for key in keys}


def compute_probe_identity(payload: dict[str, object]) -> str:
    _validate_execution_contract_hash(payload)
    return _sha256_bytes(_canonical_bytes(_probe_identity_payload(payload)))


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(_canonical_bytes(list(value.shape)) + b"\0")
    digest.update(value.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _optional_tensor_sha256(tensor: Any | None) -> str | None:
    return None if tensor is None else _tensor_sha256(tensor)


def _ordered_tensor_hashes(
    values: list[tuple[str, Any]],
) -> dict[str, str]:
    return {name: _tensor_sha256(value) for name, value in values}


def _encoding_hash(encoding: Any) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "token_ids": _tensor_sha256(encoding.token_ids),
                "segment_ids": _tensor_sha256(encoding.segment_ids),
                "argument_position_ids": _tensor_sha256(
                    encoding.argument_position_ids
                ),
                "attention_mask": _tensor_sha256(encoding.attention_mask),
                "ref_slot_positions": list(encoding.ref_slot_positions),
            }
        )
    )


def _loss_components(
    *,
    logits: Any,
    action: Any,
    arg1: Any,
    arg2: Any,
    valid: int,
    modules: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    functional = modules["functional"]
    torch_module = modules["torch"]
    actions = modules["actions"]
    flat = action[:, :valid].flatten()
    loss = functional.cross_entropy(
        logits.action[:, :valid].flatten(0, 1), flat
    )
    action_component = loss.detach().clone()
    arg1_component = torch_module.zeros((), dtype=loss.dtype)
    arg2_component = torch_module.zeros((), dtype=loss.dtype)
    one = flat != actions["END"]
    two = (flat == actions["UNSTACK"]) | (flat == actions["STACK"])
    if one.any():
        arg1_component = functional.cross_entropy(
            logits.arg1[:, :valid].flatten(0, 1)[one],
            arg1[:, :valid].flatten()[one],
        )
        loss += arg1_component
    if two.any():
        arg2_component = functional.cross_entropy(
            logits.arg2[:, :valid].flatten(0, 1)[two],
            arg2[:, :valid].flatten()[two],
        )
        loss += arg2_component
    return loss, {
        "action": action_component,
        "arg1": arg1_component,
        "arg2": arg2_component,
        "total": loss,
    }


def _optimizer_moments(
    optimizer: Any,
    named_parameters: list[tuple[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    exp_avg: dict[str, str] = {}
    exp_avg_sq: dict[str, str] = {}
    for name, parameter in named_parameters:
        state = optimizer.state.get(parameter)
        if not state:
            continue
        exp_avg[name] = _tensor_sha256(state["exp_avg"])
        exp_avg_sq[name] = _tensor_sha256(state["exp_avg_sq"])
    return exp_avg, exp_avg_sq


def _trace_update(
    *,
    model: Any,
    named_parameters: list[tuple[str, Any]],
    optimizer: Any,
    row: dict[str, object],
    update: int,
    epoch: int,
    modules: dict[str, Any],
    quality_clip_parameter_order: bool = False,
) -> dict[str, object]:
    torch_module = modules["torch"]
    action, arg1, arg2 = modules["labels"](row)
    valid = len(row["oracle_work_plan"])
    target = modules["targets"](row)
    _shifted = torch_module.cat(
        [torch_module.zeros_like(target[:, :1]), target[:, :-1]], dim=1
    )
    encoded = modules["canonical_task_encoding"](row)
    optimizer.zero_grad(set_to_none=True)
    logits = model(
        encoded,
        action,
        arg1,
        arg2,
        semantic_feedback=None,
    )
    loss, components = _loss_components(
        logits=logits,
        action=action,
        arg1=arg1,
        arg2=arg2,
        valid=valid,
        modules=modules,
    )
    forward_logits = {
        "action": _tensor_sha256(logits.action),
        "arg1": _tensor_sha256(logits.arg1),
        "arg2": _tensor_sha256(logits.arg2),
        "z_semantic": _optional_tensor_sha256(logits.z_semantic),
    }
    loss.backward()
    raw_gradients = {
        name: _tensor_sha256(parameter.grad)
        for name, parameter in named_parameters
        if parameter.grad is not None
    }
    clip_parameters = (
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if quality_clip_parameter_order
        else [parameter for _, parameter in named_parameters]
    )
    gradient_norm = torch_module.nn.utils.clip_grad_norm_(
        clip_parameters,
        1.0,
    )
    clipped_gradients = {
        name: _tensor_sha256(parameter.grad)
        for name, parameter in named_parameters
        if parameter.grad is not None
    }
    optimizer.step()
    exp_avg, exp_avg_sq = _optimizer_moments(
        optimizer,
        named_parameters,
    )
    return {
        "update": update,
        "epoch": epoch,
        "task_id": row["task_id"],
        "encoded_task_sha256": _encoding_hash(encoded),
        "forward_logits": forward_logits,
        "loss_components": {
            name: _tensor_sha256(value)
            for name, value in sorted(components.items())
        },
        "raw_gradients": raw_gradients,
        "gradient_norm": _tensor_sha256(gradient_norm),
        "gradients_after_clipping": clipped_gradients,
        "adamw_exp_avg": exp_avg,
        "adamw_exp_avg_sq": exp_avg_sq,
        "parameters_after_optimizer_step": _ordered_tensor_hashes(
            list(model.named_parameters())
        ),
    }


def _run_probe_after_preflight(
    *,
    spec: dict[str, object],
    seed: int,
) -> dict[str, object]:
    runtime, modules = _load_modules(seed)
    rows = sorted(
        modules["generate"](17)["train"],
        key=lambda item: item["task_id"],
    )
    model = modules["LockedPlanner"](seed, "A2").cpu()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    optimizer = _build_optimizer(
        modules["torch"],
        [parameter for _, parameter in named_parameters],
        spec,
    )
    contract = _execution_contract(
        spec=spec,
        optimizer=optimizer,
        torch_module=modules["torch"],
    )
    payload: dict[str, object] = {
        "probe_version": PROBE_VERSION,
        "variant": "A2",
        "seed": seed,
        "epochs": EPOCHS,
        "ordered_train_task_ids": [row["task_id"] for row in rows],
        "parameter_names": [name for name, _ in named_parameters],
        "initial_parameters": _ordered_tensor_hashes(
            list(model.named_parameters())
        ),
        "updates": [],
        "execution_contract": contract,
        "execution_contract_sha256": _sha256_bytes(
            _canonical_bytes(contract)
        ),
        "runtime": runtime,
    }
    update_index = 0
    for epoch in range(1, EPOCHS + 1):
        for row in rows:
            update_index += 1
            payload["updates"].append(
                _trace_update(
                    model=model,
                    named_parameters=named_parameters,
                    optimizer=optimizer,
                    row=row,
                    update=update_index,
                    epoch=epoch,
                    modules=modules,
                )
            )
    hardware = modules["full_hardware_runtime_fingerprint"](runtime)
    modules["validate_hardware_runtime_fingerprint"](hardware)
    payload["hardware_runtime_fingerprint"] = hardware
    payload["probe_identity"] = compute_probe_identity(payload)
    payload["evidence_identity"] = _sha256_bytes(
        _canonical_bytes(payload)
    )
    return payload


def run_probe(
    *,
    profile: str,
    seed: int = 17,
    optimizer_foreach: bool | None = None,
    optimizer_fused: bool | None = None,
) -> dict[str, object]:
    _ensure_torch_not_imported()
    spec = _resolve_profile_before_torch_import(
        profile,
        optimizer_foreach=optimizer_foreach,
        optimizer_fused=optimizer_fused,
    )
    return _run_probe_after_preflight(spec=spec, seed=seed)


def _first_mapping_difference(
    left: dict[str, object],
    right: dict[str, object],
) -> str | None:
    for name in sorted(set(left) | set(right)):
        if left.get(name) != right.get(name):
            return name
    return None


def compare_probes(
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    left_contract = _validate_execution_contract_hash(left)
    right_contract = _validate_execution_contract_hash(right)
    base: dict[str, object] = {
        "comparison_version": COMPARISON_VERSION,
        "left_probe_identity": left.get("probe_identity"),
        "right_probe_identity": right.get("probe_identity"),
        "left_execution_contract_sha256": (
            left["execution_contract_sha256"]
        ),
        "right_execution_contract_sha256": (
            right["execution_contract_sha256"]
        ),
    }
    if left_contract != right_contract:
        return {
            **base,
            "comparable": False,
            "reason": "EXECUTION_CONTRACT_MISMATCH",
            "equal": None,
            "first_divergence": None,
            "first_parameter_divergence": None,
        }
    stages = (
        "encoded_task_sha256",
        "forward_logits",
        "loss_components",
        "raw_gradients",
        "gradient_norm",
        "gradients_after_clipping",
        "adamw_exp_avg",
        "adamw_exp_avg_sq",
        "parameters_after_optimizer_step",
    )
    first = None
    first_parameter = None
    if left.get("parameter_names") != right.get("parameter_names"):
        first = {"stage": "parameter_names"}
        first_parameter = first
    elif left["initial_parameters"] != right["initial_parameters"]:
        name = _first_mapping_difference(
            left["initial_parameters"],
            right["initial_parameters"],
        )
        first = {"stage": "initial_parameters", "parameter": name}
        first_parameter = first
    elif len(left["updates"]) != len(right["updates"]):
        first = {"stage": "update_count"}
    else:
        for left_update, right_update in zip(
            left["updates"],
            right["updates"],
            strict=True,
        ):
            for stage in stages:
                if left_update[stage] == right_update[stage]:
                    continue
                first = {
                    "update": left_update["update"],
                    "epoch": left_update["epoch"],
                    "task_id": left_update["task_id"],
                    "stage": stage,
                }
                if isinstance(left_update[stage], dict):
                    first["name"] = _first_mapping_difference(
                        left_update[stage],
                        right_update[stage],
                    )
                break
            if first is not None:
                break
        for left_update, right_update in zip(
            left["updates"],
            right["updates"],
            strict=True,
        ):
            for stage in (
                "raw_gradients",
                "gradients_after_clipping",
                "adamw_exp_avg",
                "adamw_exp_avg_sq",
                "parameters_after_optimizer_step",
            ):
                if left_update[stage] == right_update[stage]:
                    continue
                first_parameter = {
                    "update": left_update["update"],
                    "epoch": left_update["epoch"],
                    "task_id": left_update["task_id"],
                    "stage": stage,
                    "parameter": _first_mapping_difference(
                        left_update[stage],
                        right_update[stage],
                    ),
                }
                break
            if first_parameter is not None:
                break
    return {
        **base,
        "comparable": True,
        "reason": None,
        "equal": first is None,
        "first_divergence": first,
        "first_parameter_divergence": first_parameter,
    }


def _assert_quality_training_source_contract() -> None:
    from planner_toy import quality

    source = inspect.getsource(quality._train)
    required = (
        "optimizer_named_parameters = _optimizer_named_parameters(model)",
        "[parameter for _, parameter in optimizer_named_parameters], lr=3e-4,",
        "betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,",
        "optimizer.zero_grad(set_to_none=True)",
        "logits = model(",
        "loss = F.cross_entropy(logits.action[:, :valid].flatten(0, 1), flat)",
        "loss += F.cross_entropy(logits.arg1[:, :valid].flatten(0, 1)[one]",
        "loss += F.cross_entropy(logits.arg2[:, :valid].flatten(0, 1)[two]",
        "loss.backward()",
        "torch.nn.utils.clip_grad_norm_("
        "[p for p in model.parameters() if p.requires_grad], 1.0)",
        "optimizer.step()",
    )
    for fragment in required:
        if fragment not in source:
            raise RuntimeError(
                f"QUALITY_TRAINING_SOURCE_CONTRACT_DRIFT:{fragment}"
            )
    ordered = (
        "optimizer.zero_grad(set_to_none=True)",
        "logits = model(",
        "loss = F.cross_entropy",
        "loss.backward()",
        "torch.nn.utils.clip_grad_norm_",
        "optimizer.step()",
    )
    positions = [source.index(fragment) for fragment in ordered]
    if positions != sorted(positions):
        raise RuntimeError("QUALITY_TRAINING_UPDATE_ORDER_DRIFT")


def _one_update_trace(
    *,
    seed: int,
    row: dict[str, object],
    modules: dict[str, Any],
    named_parameter_factory: Any,
    spec: dict[str, object],
    quality_clip_parameter_order: bool,
) -> dict[str, object]:
    modules["configure"](seed)
    model = modules["LockedPlanner"](seed, "A2").cpu()
    named_parameters = named_parameter_factory(model)
    optimizer = _build_optimizer(
        modules["torch"],
        [parameter for _, parameter in named_parameters],
        spec,
    )
    initial = _ordered_tensor_hashes(list(model.named_parameters()))
    update = _trace_update(
        model=model,
        named_parameters=named_parameters,
        optimizer=optimizer,
        row=row,
        update=1,
        epoch=1,
        modules=modules,
        quality_clip_parameter_order=quality_clip_parameter_order,
    )
    return {
        "parameter_names": [name for name, _ in named_parameters],
        "optimizer_defaults": _normalize_value(optimizer.defaults),
        "initial_parameters": initial,
        **{
            key: update[key]
            for key in (
                "encoded_task_sha256",
                "forward_logits",
                "loss_components",
                "raw_gradients",
                "gradient_norm",
                "gradients_after_clipping",
                "adamw_exp_avg",
                "adamw_exp_avg_sq",
                "parameters_after_optimizer_step",
            )
        },
    }


def run_quality_training_parity(
    *,
    seed: int = 17,
) -> dict[str, object]:
    _ensure_torch_not_imported()
    spec = _resolve_profile_before_torch_import("historical-default")
    runtime, modules = _load_modules(seed)
    del runtime
    from planner_toy import quality

    _assert_quality_training_source_contract()
    row = sorted(
        modules["generate"](17)["train"],
        key=lambda item: item["task_id"],
    )[0]
    quality_trace = _one_update_trace(
        seed=seed,
        row=row,
        modules=modules,
        named_parameter_factory=quality._optimizer_named_parameters,
        spec=spec,
        quality_clip_parameter_order=True,
    )
    probe_trace = _one_update_trace(
        seed=seed,
        row=row,
        modules=modules,
        named_parameter_factory=lambda model: [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        spec=spec,
        quality_clip_parameter_order=False,
    )
    modules["configure"](seed)
    model = modules["LockedPlanner"](seed, "A2").cpu()
    contract_optimizer = _build_optimizer(
        modules["torch"],
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        spec,
    )
    contract = _execution_contract(
        spec=spec,
        optimizer=contract_optimizer,
        torch_module=modules["torch"],
    )
    return {
        "parity_version": PARITY_VERSION,
        "profile": "historical-default",
        "execution_contract": contract,
        "task_id": row["task_id"],
        "equal": quality_trace == probe_trace,
        "quality_trace": quality_trace,
        "probe_trace": probe_trace,
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--profile",
        choices=tuple(_PROFILE_SPECS),
        required=True,
    )
    run_parser.add_argument("--optimizer-foreach", type=_parse_bool)
    run_parser.add_argument("--optimizer-fused", type=_parse_bool)
    run_parser.add_argument("--seed", type=int, default=17)
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument(
        "--expect",
        choices=("equal", "different", "incomparable"),
    )
    parity_parser = subparsers.add_parser("parity")
    parity_parser.add_argument("--seed", type=int, default=17)
    parity_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_probe(
            profile=args.profile,
            seed=args.seed,
            optimizer_foreach=args.optimizer_foreach,
            optimizer_fused=args.optimizer_fused,
        )
        output = args.output
    elif args.command == "parity":
        result = run_quality_training_parity(seed=args.seed)
        output = args.output
        if result["equal"] is not True:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(_canonical_bytes(result) + b"\n")
            raise SystemExit("quality training parity failed")
    else:
        result = compare_probes(
            _read_json(args.left),
            _read_json(args.right),
        )
        output = args.output
        if args.expect is not None:
            observed = (
                "incomparable"
                if result["comparable"] is False
                else "equal"
                if result["equal"] is True
                else "different"
            )
            if observed != args.expect:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(_canonical_bytes(result) + b"\n")
                raise SystemExit(
                    "probe comparison expectation failed:"
                    f"expected={args.expect}:actual={observed}"
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(result) + b"\n")


if __name__ == "__main__":
    main()
