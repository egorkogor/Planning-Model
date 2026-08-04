"""Update-level deterministic probe for canonical A2 CPU training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from planner_toy.canonical_runtime import (
    CANONICAL_CPU_RUNTIME_VERSION as LEGACY_CANONICAL_CPU_RUNTIME_VERSION,
    configure_canonical_cpu_runtime,
)
from planner_toy.dataset import generate
from planner_toy.model import LockedPlanner, TaskEncoding, canonical_task_encoding
from planner_toy.semantic import targets
from planner_toy.training import ACTIONS, labels
from scripts.canonical_cpu_hardware_fingerprint import (
    full_hardware_runtime_fingerprint,
)

PROBE_VERSION = "toy-quality-canonical-training-probe/1.0"
EPOCHS = 3


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(_canonical_bytes(list(value.shape)) + b"\0")
    digest.update(value.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _optional_tensor_sha256(tensor: torch.Tensor | None) -> str | None:
    return None if tensor is None else _tensor_sha256(tensor)


def _tensor_map_hash(values: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: _tensor_sha256(values[name]) for name in sorted(values)}


def _encoding_hash(encoding: TaskEncoding) -> str:
    payload = {
        "token_ids": _tensor_sha256(encoding.token_ids),
        "segment_ids": _tensor_sha256(encoding.segment_ids),
        "argument_position_ids": _tensor_sha256(encoding.argument_position_ids),
        "attention_mask": _tensor_sha256(encoding.attention_mask),
        "ref_slot_positions": list(encoding.ref_slot_positions),
    }
    return _sha256_bytes(_canonical_bytes(payload))


def _configure_legacy(seed: int) -> dict[str, object]:
    for name in ("ATEN_CPU_CAPABILITY", "MKL_CBWR"):
        if os.environ.get(name) is not None:
            raise RuntimeError(f"LEGACY_PROBE_ENVIRONMENT_CONTAMINATED:{name}")
    expected_threads = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for name, expected in expected_threads.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"LEGACY_PROBE_ENVIRONMENT_MISMATCH:{name}")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        raise RuntimeError("LEGACY_PROBE_CONFIGURATION_LATE") from error
    torch.backends.mkldnn.enabled = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return {
        "profile_version": LEGACY_CANONICAL_CPU_RUNTIME_VERSION,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only_enabled": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        **expected_threads,
    }


def _loss_components(
    logits: Any,
    action: torch.Tensor,
    arg1: torch.Tensor,
    arg2: torch.Tensor,
    valid: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    flat = action[:, :valid].flatten()
    loss = F.cross_entropy(logits.action[:, :valid].flatten(0, 1), flat)
    action_component = loss.detach().clone()
    arg1_component = torch.zeros((), dtype=loss.dtype)
    arg2_component = torch.zeros((), dtype=loss.dtype)
    one = flat != ACTIONS["END"]
    two = (flat == ACTIONS["UNSTACK"]) | (flat == ACTIONS["STACK"])
    if one.any():
        arg1_component = F.cross_entropy(
            logits.arg1[:, :valid].flatten(0, 1)[one],
            arg1[:, :valid].flatten()[one],
        )
        loss += arg1_component
    if two.any():
        arg2_component = F.cross_entropy(
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


def _operator_trace(row: dict, seed: int) -> list[str]:
    torch.manual_seed(seed)
    model = LockedPlanner(seed, "A2").cpu()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    action, arg1, arg2 = labels(row)
    valid = len(row["oracle_work_plan"])
    encoded = canonical_task_encoding(row)
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        optimizer.zero_grad(set_to_none=True)
        logits = model(encoded, action, arg1, arg2, semantic_feedback=None)
        loss, _ = _loss_components(logits, action, arg1, arg2, valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
    return sorted({event.key for event in profiler.key_averages()})


def run_probe(*, mode: str, seed: int = 17) -> dict[str, object]:
    if mode == "canonical":
        runtime = configure_canonical_cpu_runtime(seed)
    elif mode == "legacy":
        runtime = _configure_legacy(seed)
    else:
        raise ValueError(f"unsupported probe mode: {mode}")
    hardware = full_hardware_runtime_fingerprint(runtime)

    dataset = generate(17)
    rows = sorted(dataset["train"], key=lambda item: item["task_id"])
    model = LockedPlanner(seed, "A2").cpu()
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    initial_parameters = _tensor_map_hash(
        {name: parameter for name, parameter in model.named_parameters()}
    )
    updates: list[dict[str, object]] = []
    update_index = 0
    for epoch in range(EPOCHS):
        for row in rows:
            update_index += 1
            action, arg1, arg2 = labels(row)
            valid = len(row["oracle_work_plan"])
            target = targets(row)
            _shifted = torch.cat(
                [torch.zeros_like(target[:, :1]), target[:, :-1]], dim=1
            )
            encoded = canonical_task_encoding(row)
            optimizer.zero_grad(set_to_none=True)
            logits = model(encoded, action, arg1, arg2, semantic_feedback=None)
            loss, components = _loss_components(logits, action, arg1, arg2, valid)
            forward_logits = {
                "action": _tensor_sha256(logits.action),
                "arg1": _tensor_sha256(logits.arg1),
                "arg2": _tensor_sha256(logits.arg2),
                "z_semantic": _optional_tensor_sha256(logits.z_semantic),
            }
            loss.backward()
            raw_gradients = {
                name: _tensor_sha256(parameter.grad)
                for name, parameter in named
                if parameter.grad is not None
            }
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named], 1.0
            )
            clipped_gradients = {
                name: _tensor_sha256(parameter.grad)
                for name, parameter in named
                if parameter.grad is not None
            }
            optimizer.step()
            exp_avg: dict[str, str] = {}
            exp_avg_sq: dict[str, str] = {}
            for name, parameter in named:
                state = optimizer.state.get(parameter)
                if not state:
                    continue
                exp_avg[name] = _tensor_sha256(state["exp_avg"])
                exp_avg_sq[name] = _tensor_sha256(state["exp_avg_sq"])
            parameters_after_step = _tensor_map_hash(
                {name: parameter for name, parameter in model.named_parameters()}
            )
            updates.append(
                {
                    "update": update_index,
                    "epoch": epoch + 1,
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
                    "parameters_after_optimizer_step": parameters_after_step,
                }
            )
    numerical_evidence = {
        "probe_version": PROBE_VERSION,
        "variant": "A2",
        "seed": seed,
        "epochs": EPOCHS,
        "ordered_train_task_ids": [row["task_id"] for row in rows],
        "initial_parameters": initial_parameters,
        "updates": updates,
    }
    payload: dict[str, object] = {
        **numerical_evidence,
        "mode": mode,
        "runtime": runtime,
        "hardware_runtime_fingerprint": hardware,
        "operator_trace": _operator_trace(rows[0], seed),
        "probe_identity": _sha256_bytes(_canonical_bytes(numerical_evidence)),
    }
    payload["evidence_identity"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _first_mapping_difference(
    left: dict[str, object], right: dict[str, object]
) -> str | None:
    for name in sorted(set(left) | set(right)):
        if left.get(name) != right.get(name):
            return name
    return None


def _hardware_identity(payload: dict[str, object]) -> object:
    fingerprint = payload["hardware_runtime_fingerprint"]
    if not isinstance(fingerprint, dict):
        return None
    return fingerprint.get(
        "semantic_execution_identity_sha256",
        fingerprint.get("observation_identity_sha256"),
    )


def compare_probes(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
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
    first_divergence = None
    first_parameter_divergence = None
    if left["initial_parameters"] != right["initial_parameters"]:
        name = _first_mapping_difference(
            left["initial_parameters"], right["initial_parameters"]
        )
        first_divergence = {"stage": "initial_parameters", "parameter": name}
        first_parameter_divergence = first_divergence
    else:
        left_updates = left["updates"]
        right_updates = right["updates"]
        if len(left_updates) != len(right_updates):
            first_divergence = {"stage": "update_count"}
        else:
            for left_update, right_update in zip(
                left_updates, right_updates, strict=True
            ):
                for stage in stages:
                    if left_update[stage] == right_update[stage]:
                        continue
                    detail: dict[str, object] = {
                        "update": left_update["update"],
                        "epoch": left_update["epoch"],
                        "task_id": left_update["task_id"],
                        "stage": stage,
                    }
                    if isinstance(left_update[stage], dict):
                        detail["name"] = _first_mapping_difference(
                            left_update[stage], right_update[stage]
                        )
                    first_divergence = detail
                    break
                if first_divergence is not None:
                    break
            for left_update, right_update in zip(
                left_updates, right_updates, strict=True
            ):
                for stage in (
                    "raw_gradients",
                    "gradients_after_clipping",
                    "adamw_exp_avg",
                    "adamw_exp_avg_sq",
                    "parameters_after_optimizer_step",
                ):
                    if left_update[stage] != right_update[stage]:
                        first_parameter_divergence = {
                            "update": left_update["update"],
                            "epoch": left_update["epoch"],
                            "task_id": left_update["task_id"],
                            "stage": stage,
                            "parameter": _first_mapping_difference(
                                left_update[stage], right_update[stage]
                            ),
                        }
                        break
                if first_parameter_divergence is not None:
                    break
    return {
        "comparison_version": (
            "toy-quality-canonical-training-probe-comparison/1.0"
        ),
        "equal": first_divergence is None,
        "left_probe_identity": left["probe_identity"],
        "right_probe_identity": right["probe_identity"],
        "first_divergence": first_divergence,
        "first_parameter_divergence": first_parameter_divergence,
        "left_semantic_execution_identity": _hardware_identity(left),
        "right_semantic_execution_identity": _hardware_identity(right),
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mode", choices=("legacy", "canonical"), required=True)
    run_parser.add_argument("--seed", type=int, default=17)
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--expect", choices=("equal", "different"))
    args = parser.parse_args()
    if args.command == "run":
        result = run_probe(mode=args.mode, seed=args.seed)
    else:
        result = compare_probes(_read_json(args.left), _read_json(args.right))
        if args.expect is not None:
            expected = args.expect == "equal"
            if result["equal"] != expected:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes(_canonical_bytes(result) + b"\n")
                raise SystemExit(
                    f"probe comparison expectation failed: expected {args.expect}"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(result) + b"\n")


if __name__ == "__main__":
    main()
