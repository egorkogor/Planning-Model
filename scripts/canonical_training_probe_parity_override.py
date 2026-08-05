"""Order-independent adapter for transparent quality-training parity capture."""
from __future__ import annotations

from typing import Any

from scripts.canonical_training_probe_contract import (
    PARITY_VERSION,
    _build_optimizer,
    _ensure_torch_not_imported,
    _execution_contract,
    _load_modules,
    _resolve_profile_before_torch_import,
)
from scripts.canonical_training_probe_parity import (
    _first_parity_difference,
    _instrument_quality_training_update,
    _run_probe_one_update,
)


def _instrument_with_eager_parameter_capture(
    *, seed: int, row: dict[str, object], modules: dict[str, Any]
) -> dict[str, object]:
    """Ensure the real quality path exposes optimizer order before backward hooks.

    The underlying instrumentation remains responsible for all captured values.
    This adapter only makes the parameter-selection call order-independent by
    invoking the currently installed quality helper when the model is created.
    """
    from planner_toy import quality

    original_locked_planner = quality.LockedPlanner

    def eager_locked_planner(*args: Any, **kwargs: Any) -> Any:
        model = original_locked_planner(*args, **kwargs)
        quality._optimizer_named_parameters(model)
        return model

    quality.LockedPlanner = eager_locked_planner
    try:
        return _instrument_quality_training_update(
            seed=seed,
            row=row,
            modules=modules,
        )
    finally:
        quality.LockedPlanner = original_locked_planner


def run_quality_training_parity(*, seed: int = 17) -> dict[str, object]:
    _ensure_torch_not_imported()
    spec = _resolve_profile_before_torch_import("historical-default")
    runtime, modules = _load_modules(seed)
    del runtime
    row = sorted(
        modules["generate"](17)["train"], key=lambda item: item["task_id"]
    )[0]
    quality_trace = _instrument_with_eager_parameter_capture(
        seed=seed,
        row=row,
        modules=modules,
    )
    probe_trace = _run_probe_one_update(
        seed=seed,
        row=row,
        modules=modules,
        spec=spec,
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
