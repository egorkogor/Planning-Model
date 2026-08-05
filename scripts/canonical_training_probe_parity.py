"""Transparent one-update parity instrumentation for frozen quality training."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from scripts.canonical_training_probe_contract import (
    PARITY_VERSION,
    _build_optimizer,
    _ensure_torch_not_imported,
    _execution_contract,
    _load_modules,
    _normalize_value,
    _resolve_profile_before_torch_import,
)
from scripts.canonical_training_probe_core import (
    _encoding_hash,
    _optional_tensor_sha256,
    _optimizer_moments,
    _ordered_tensor_hashes,
    _tensor_sha256,
    _trace_update,
)

def _parity_trace_from_probe_update(
    *,
    model: Any,
    named_parameters: list[tuple[str, Any]],
    optimizer: Any,
    update: dict[str, object],
) -> dict[str, object]:
    clip_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    return {
        "parameter_names": [name for name, _ in named_parameters],
        "optimizer_defaults": _normalize_value(optimizer.defaults),
        "initial_parameters": _ordered_tensor_hashes(list(model.named_parameters())),
        "encoded_task_sha256": update["encoded_task_sha256"],
        "forward_logits": update["forward_logits"],
        "loss_components": update["loss_components"],
        "raw_gradients": update["raw_gradients"],
        "gradient_norm": update["gradient_norm"],
        "gradients_after_clipping": update["gradients_after_clipping"],
        "adamw_exp_avg": update["adamw_exp_avg"],
        "adamw_exp_avg_sq": update["adamw_exp_avg_sq"],
        "parameters_after_optimizer_step": update[
            "parameters_after_optimizer_step"
        ],
        "gradient_clip_max_norm": 1.0,
        "gradient_clip_parameter_names": clip_names,
        "update_events": ["zero_grad", "forward", "loss", "backward", "clip", "step"],
    }


def _run_probe_one_update(
    *,
    seed: int,
    row: dict[str, object],
    modules: dict[str, Any],
    spec: dict[str, object],
) -> dict[str, object]:
    modules["configure"](seed)
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
    initial_parameters = _ordered_tensor_hashes(list(model.named_parameters()))
    update = _trace_update(
        model=model,
        named_parameters=named_parameters,
        optimizer=optimizer,
        row=row,
        update=1,
        epoch=1,
        modules=modules,
    )
    trace = _parity_trace_from_probe_update(
        model=model,
        named_parameters=named_parameters,
        optimizer=optimizer,
        update=update,
    )
    trace["initial_parameters"] = initial_parameters
    return trace


def _parameter_names_for_objects(model: Any, values: list[Any]) -> list[str]:
    by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        return [by_identity[id(value)] for value in values]
    except KeyError as error:
        raise RuntimeError("QUALITY_PARITY_UNKNOWN_PARAMETER") from error


def _instrument_quality_training_update(
    *,
    seed: int,
    row: dict[str, object],
    modules: dict[str, Any],
) -> dict[str, object]:
    from planner_toy import quality

    torch_module = modules["torch"]
    original_epochs = quality.EPOCHS
    original_locked_planner = quality.LockedPlanner
    original_optimizer_named_parameters = quality._optimizer_named_parameters
    original_adamw = quality.torch.optim.AdamW
    original_cross_entropy = quality.F.cross_entropy
    original_clip_grad_norm = quality.torch.nn.utils.clip_grad_norm_
    original_backward = torch_module.Tensor.backward
    captured: dict[str, Any] = {
        "events": [],
        "cross_entropy_components": [],
    }

    def locked_planner_factory(*args: Any, **kwargs: Any) -> Any:
        model = original_locked_planner(*args, **kwargs)
        captured["model"] = model
        captured["initial_parameters"] = _ordered_tensor_hashes(
            list(model.named_parameters())
        )
        original_forward = model.forward

        def forward(*forward_args: Any, **forward_kwargs: Any) -> Any:
            captured["events"].append("forward")
            captured["encoded_task_sha256"] = _encoding_hash(forward_args[0])
            logits = original_forward(*forward_args, **forward_kwargs)
            captured["forward_logits"] = {
                "action": _tensor_sha256(logits.action),
                "arg1": _tensor_sha256(logits.arg1),
                "arg2": _tensor_sha256(logits.arg2),
                "z_semantic": _optional_tensor_sha256(logits.z_semantic),
            }
            return logits

        model.forward = forward
        return model

    def optimizer_named_parameters(model: Any) -> list[tuple[str, Any]]:
        values = original_optimizer_named_parameters(model)
        if "named_parameters" not in captured:
            captured["named_parameters"] = values
            captured["parameter_names"] = [name for name, _ in values]
        return values

    def adamw_factory(parameters: Any, *args: Any, **kwargs: Any) -> Any:
        parameter_list = list(parameters)
        optimizer = original_adamw(parameter_list, *args, **kwargs)
        captured["optimizer"] = optimizer
        captured["optimizer_defaults"] = _normalize_value(optimizer.defaults)
        captured["optimizer_parameter_names"] = _parameter_names_for_objects(
            captured["model"], parameter_list
        )
        original_zero_grad = optimizer.zero_grad
        original_step = optimizer.step

        def zero_grad(*zero_args: Any, **zero_kwargs: Any) -> Any:
            captured["events"].append("zero_grad")
            return original_zero_grad(*zero_args, **zero_kwargs)

        def step(*step_args: Any, **step_kwargs: Any) -> Any:
            captured["events"].append("step")
            result = original_step(*step_args, **step_kwargs)
            exp_avg, exp_avg_sq = _optimizer_moments(
                optimizer, captured["named_parameters"]
            )
            captured["adamw_exp_avg"] = exp_avg
            captured["adamw_exp_avg_sq"] = exp_avg_sq
            captured["parameters_after_optimizer_step"] = _ordered_tensor_hashes(
                list(captured["model"].named_parameters())
            )
            return result

        optimizer.zero_grad = zero_grad
        optimizer.step = step
        return optimizer

    def cross_entropy(*args: Any, **kwargs: Any) -> Any:
        result = original_cross_entropy(*args, **kwargs)
        captured["cross_entropy_components"].append(result.detach().clone())
        return result

    def backward(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        captured["events"].append("loss")
        captured["total_loss"] = tensor.detach().clone()
        result = original_backward(tensor, *args, **kwargs)
        captured["events"].append("backward")
        captured["raw_gradients"] = {
            name: _tensor_sha256(parameter.grad)
            for name, parameter in captured["named_parameters"]
            if parameter.grad is not None
        }
        return result

    def clip_grad_norm_(
        parameters: Any, max_norm: float, *args: Any, **kwargs: Any
    ) -> Any:
        parameter_list = list(parameters)
        captured["events"].append("clip")
        captured["gradient_clip_max_norm"] = float(max_norm)
        captured["gradient_clip_parameter_names"] = _parameter_names_for_objects(
            captured["model"], parameter_list
        )
        result = original_clip_grad_norm(
            parameter_list, max_norm, *args, **kwargs
        )
        captured["gradient_norm"] = _tensor_sha256(result)
        captured["gradients_after_clipping"] = {
            name: _tensor_sha256(parameter.grad)
            for name, parameter in captured["named_parameters"]
            if parameter.grad is not None
        }
        return result

    quality.EPOCHS = 1
    quality.LockedPlanner = locked_planner_factory
    quality._optimizer_named_parameters = optimizer_named_parameters
    quality.torch.optim.AdamW = adamw_factory
    quality.F.cross_entropy = cross_entropy
    quality.torch.nn.utils.clip_grad_norm_ = clip_grad_norm_
    torch_module.Tensor.backward = backward
    try:
        with tempfile.TemporaryDirectory(prefix="quality-parity-") as directory:
            quality._train(
                [row],
                "A2",
                seed,
                Path(directory),
                "sha256:" + "0" * 64,
            )
    finally:
        quality.EPOCHS = original_epochs
        quality.LockedPlanner = original_locked_planner
        quality._optimizer_named_parameters = original_optimizer_named_parameters
        quality.torch.optim.AdamW = original_adamw
        quality.F.cross_entropy = original_cross_entropy
        quality.torch.nn.utils.clip_grad_norm_ = original_clip_grad_norm
        torch_module.Tensor.backward = original_backward

    required = {
        "model",
        "initial_parameters",
        "named_parameters",
        "parameter_names",
        "optimizer",
        "optimizer_defaults",
        "optimizer_parameter_names",
        "encoded_task_sha256",
        "forward_logits",
        "total_loss",
        "raw_gradients",
        "gradient_clip_max_norm",
        "gradient_clip_parameter_names",
        "gradient_norm",
        "gradients_after_clipping",
        "adamw_exp_avg",
        "adamw_exp_avg_sq",
        "parameters_after_optimizer_step",
    }
    missing = sorted(required - set(captured))
    if missing:
        raise RuntimeError(f"QUALITY_PARITY_INSTRUMENTATION_INCOMPLETE:{missing}")
    if captured["optimizer_parameter_names"] != captured["parameter_names"]:
        raise RuntimeError("QUALITY_PARITY_OPTIMIZER_PARAMETER_ORDER_MISMATCH")
    action, _, _ = modules["labels"](row)
    valid = len(row["oracle_work_plan"])
    flat = action[:, :valid].flatten()
    one = bool((flat != modules["actions"]["END"]).any().item())
    two = bool(
        (
            (flat == modules["actions"]["UNSTACK"])
            | (flat == modules["actions"]["STACK"])
        ).any().item()
    )
    components = captured["cross_entropy_components"]
    expected_component_count = 1 + int(one) + int(two)
    if len(components) != expected_component_count:
        raise RuntimeError("QUALITY_PARITY_LOSS_COMPONENT_COUNT_MISMATCH")
    cursor = 1
    zero = torch_module.zeros((), dtype=components[0].dtype)
    arg1_component = components[cursor] if one else zero
    cursor += int(one)
    arg2_component = components[cursor] if two else zero
    loss_components = {
        "action": _tensor_sha256(components[0]),
        "arg1": _tensor_sha256(arg1_component),
        "arg2": _tensor_sha256(arg2_component),
        "total": _tensor_sha256(captured["total_loss"]),
    }
    return {
        "parameter_names": captured["parameter_names"],
        "optimizer_defaults": captured["optimizer_defaults"],
        "initial_parameters": captured["initial_parameters"],
        "encoded_task_sha256": captured["encoded_task_sha256"],
        "forward_logits": captured["forward_logits"],
        "loss_components": loss_components,
        "raw_gradients": captured["raw_gradients"],
        "gradient_norm": captured["gradient_norm"],
        "gradients_after_clipping": captured["gradients_after_clipping"],
        "adamw_exp_avg": captured["adamw_exp_avg"],
        "adamw_exp_avg_sq": captured["adamw_exp_avg_sq"],
        "parameters_after_optimizer_step": captured[
            "parameters_after_optimizer_step"
        ],
        "gradient_clip_max_norm": captured["gradient_clip_max_norm"],
        "gradient_clip_parameter_names": captured[
            "gradient_clip_parameter_names"
        ],
        "update_events": captured["events"],
    }


def _first_parity_difference(
    quality_trace: dict[str, object], probe_trace: dict[str, object]
) -> str | None:
    for name in quality_trace:
        if quality_trace[name] != probe_trace.get(name):
            return name
    for name in probe_trace:
        if name not in quality_trace:
            return name
    return None


def run_quality_training_parity(*, seed: int = 17) -> dict[str, object]:
    _ensure_torch_not_imported()
    spec = _resolve_profile_before_torch_import("historical-default")
    runtime, modules = _load_modules(seed)
    del runtime
    row = sorted(
        modules["generate"](17)["train"], key=lambda item: item["task_id"]
    )[0]
    quality_trace = _instrument_quality_training_update(
        seed=seed, row=row, modules=modules
    )
    probe_trace = _run_probe_one_update(
        seed=seed, row=row, modules=modules, spec=spec
    )
    modules["configure"](seed)
    model = modules["LockedPlanner"](seed, "A2").cpu()
    contract_optimizer = _build_optimizer(
        modules["torch"],
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        spec,
    )
    contract = _execution_contract(
        spec=spec,
        optimizer=contract_optimizer,
        torch_module=modules["torch"],
    )
    first_difference = _first_parity_difference(quality_trace, probe_trace)
    return {
        "parity_version": PARITY_VERSION,
        "profile": "historical-default",
        "execution_contract": contract,
        "task_id": row["task_id"],
        "equal": first_difference is None,
        "first_difference": first_difference,
        "quality_trace": quality_trace,
        "probe_trace": probe_trace,
    }


