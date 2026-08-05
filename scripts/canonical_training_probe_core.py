"""Exact training probe generation and contract-aware comparison."""
from __future__ import annotations

import hashlib
from typing import Any

from scripts.canonical_training_probe_contract import (
    COMPARISON_VERSION,
    EPOCHS,
    PROBE_VERSION,
    _build_optimizer,
    _canonical_bytes,
    _ensure_torch_not_imported,
    _execution_contract,
    _load_modules,
    _resolve_profile_before_torch_import,
    _sha256_bytes,
    _validate_execution_contract_hash,
    compute_probe_identity,
    validate_probe_identity,
)

def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(_canonical_bytes(list(value.shape)) + b"\0")
    digest.update(value.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _optional_tensor_sha256(tensor: Any | None) -> str | None:
    return None if tensor is None else _tensor_sha256(tensor)


def _ordered_tensor_hashes(values: list[tuple[str, Any]]) -> dict[str, str]:
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
        [parameter for parameter in model.parameters() if parameter.requires_grad]
        if quality_clip_parameter_order
        else [parameter for _, parameter in named_parameters]
    )
    gradient_norm = torch_module.nn.utils.clip_grad_norm_(clip_parameters, 1.0)
    clipped_gradients = {
        name: _tensor_sha256(parameter.grad)
        for name, parameter in named_parameters
        if parameter.grad is not None
    }
    optimizer.step()
    exp_avg, exp_avg_sq = _optimizer_moments(optimizer, named_parameters)
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
    *, spec: dict[str, object], seed: int
) -> dict[str, object]:
    runtime, modules = _load_modules(seed)
    rows = sorted(
        modules["generate"](17)["train"], key=lambda item: item["task_id"]
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
        spec=spec, optimizer=optimizer, torch_module=modules["torch"]
    )
    payload: dict[str, object] = {
        "probe_version": PROBE_VERSION,
        "variant": "A2",
        "seed": seed,
        "epochs": EPOCHS,
        "ordered_train_task_ids": [row["task_id"] for row in rows],
        "parameter_names": [name for name, _ in named_parameters],
        "initial_parameters": _ordered_tensor_hashes(list(model.named_parameters())),
        "updates": [],
        "execution_contract": contract,
        "execution_contract_sha256": _sha256_bytes(_canonical_bytes(contract)),
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
    validate_probe_identity(payload)
    payload["evidence_identity"] = _sha256_bytes(_canonical_bytes(payload))
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
    left: dict[str, object], right: dict[str, object]
) -> str | None:
    for name in sorted(set(left) | set(right)):
        if left.get(name) != right.get(name):
            return name
    return None


def _incomparable_report(
    base: dict[str, object], reason: str
) -> dict[str, object]:
    return {
        **base,
        "comparable": False,
        "reason": reason,
        "equal": None,
        "first_divergence": None,
        "first_parameter_divergence": None,
    }


def compare_probes(
    left: dict[str, object], right: dict[str, object]
) -> dict[str, object]:
    left_contract = _validate_execution_contract_hash(left)
    right_contract = _validate_execution_contract_hash(right)
    base: dict[str, object] = {
        "comparison_version": COMPARISON_VERSION,
        "left_probe_identity": left.get("probe_identity"),
        "right_probe_identity": right.get("probe_identity"),
        "left_execution_contract_sha256": left["execution_contract_sha256"],
        "right_execution_contract_sha256": right["execution_contract_sha256"],
    }
    if left_contract != right_contract:
        return _incomparable_report(base, "EXECUTION_CONTRACT_MISMATCH")
    validate_probe_identity(left)
    validate_probe_identity(right)
    specification_fields = (
        "probe_version",
        "variant",
        "seed",
        "epochs",
        "ordered_train_task_ids",
    )
    if any(left[name] != right[name] for name in specification_fields):
        return _incomparable_report(base, "PROBE_SPECIFICATION_MISMATCH")
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
    if left["parameter_names"] != right["parameter_names"]:
        first = {"stage": "parameter_names"}
        first_parameter = first
    elif left["initial_parameters"] != right["initial_parameters"]:
        name = _first_mapping_difference(
            left["initial_parameters"], right["initial_parameters"]
        )
        first = {"stage": "initial_parameters", "parameter": name}
        first_parameter = first
    else:
        for left_update, right_update in zip(
            left["updates"], right["updates"], strict=True
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
                        left_update[stage], right_update[stage]
                    )
                break
            if first is not None:
                break
        for left_update, right_update in zip(
            left["updates"], right["updates"], strict=True
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
                        left_update[stage], right_update[stage]
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


