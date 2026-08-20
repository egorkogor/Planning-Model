"""Independent accepted-reference reconstruction for the A2 clipping experiment."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .a2_sufficient_budget_task_order_validator import (
    _free_summary,
    _frozen_control_projection,
    _teacher_summary,
)
from .canonical_runtime import configure_canonical_cpu_runtime
from .learnability import _read_only_diagnostic_pass, free_running_task, teacher_forced_task
from .model import LockedPlanner, canonical_task_encoding
from .numeric_identity import canonical_state_dict_sha256, canonical_torch_object_sha256
from .quality import _optimizer_named_parameters
from .training import ACTIONS, labels

REFERENCE_FIELDS = (
    "update_index",
    "epoch_index",
    "task_id",
    "operator_loss",
    "arg1_pointer_loss",
    "arg2_pointer_loss",
    "total_loss",
    "operator_target_count",
    "arg1_target_count",
    "arg2_target_count",
    "gradient_norm",
    "gradient_clip_norm",
    "clipping_occurred",
    "operator_position_weight",
)
TASKS = ("bw-00000001", "bw-00000002", "bw-00000003")
NONTRIVIAL = ("bw-00000002", "bw-00000003")
CHECKPOINTS = (3, 10, 30, 100)
PERSISTENCE_CHECKPOINTS = (10, 30, 100)
MAX_EPOCH = 100


def _epoch_evidence(
    model: LockedPlanner, rows: list[dict[str, Any]], *, seed: int, epoch: int
) -> dict[str, Any]:
    with _read_only_diagnostic_pass(model):
        teacher = [teacher_forced_task(model, row, split="train", seed=seed) for row in rows]
        free = [free_running_task(model, row, split="train", seed=seed) for row in rows]
    return {
        "epoch": epoch,
        "update_count": epoch * len(rows),
        "position0": [
            {
                "task_id": task["task_id"],
                "gold_operator": task["positions"][0]["gold_operator"],
                "predicted_operator": task["positions"][0]["predicted_operator"],
                "operator_correct": task["positions"][0]["operator_correct"],
                "probability_gold_operator": task["positions"][0]["probability_gold_operator"],
                "operator_nll": task["positions"][0]["operator_nll"],
                "probability_end": task["positions"][0]["probability_end"],
            }
            for task in teacher
        ],
        "free_running": [
            {
                "task_id": task["task_id"],
                "initial_goal_satisfied": task["initial_goal_satisfied"],
                "predicted_plan": task["predicted_plan"],
                "predicted_plan_length": task["predicted_plan_length"],
                "exact_plan_match": task["exact_plan_match"],
                "final_goal_success": task["final_goal_success"],
                "failure_code": task["failure_code"],
            }
            for task in free
        ],
    }


def _checkpoint(
    model: LockedPlanner,
    optimizer: torch.optim.Optimizer,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    with _read_only_diagnostic_pass(model):
        teacher = [teacher_forced_task(model, row, split="train", seed=seed) for row in rows]
        free = [free_running_task(model, row, split="train", seed=seed) for row in rows]
    return {
        "epoch": epoch,
        "update_count": epoch * len(rows),
        "trained_canonical_sha256": canonical_state_dict_sha256(model.state_dict()),
        "optimizer_canonical_sha256": canonical_torch_object_sha256(optimizer.state_dict()),
        "teacher_forced": teacher,
        "teacher_forced_summary": _teacher_summary(teacher),
        "free_running": free,
        "free_running_summary": _free_summary(free),
    }


def _position0_rescued(record: dict[str, Any]) -> bool:
    by_id = {item["task_id"]: item for item in record["position0"]}
    return all(
        by_id[task]["gold_operator"] == "UNSTACK"
        and bool(by_id[task]["operator_correct"])
        for task in NONTRIVIAL
    )


def _free_rescued(record: dict[str, Any]) -> bool:
    by_id = {item["task_id"]: item for item in record["free_running"]}
    return all(
        by_id[task]["initial_goal_satisfied"] is False
        and bool(by_id[task]["final_goal_success"])
        for task in NONTRIVIAL
    )


def _first(records: list[dict[str, Any]], predicate) -> dict[str, int] | None:
    for record in records:
        if predicate(record):
            return {"epoch": int(record["epoch"]), "update_count": int(record["update_count"])}
    return None


def _persistence(
    records: list[dict[str, Any]], event: dict[str, int] | None, predicate
) -> dict[str, bool | None]:
    by_epoch = {int(record["epoch"]): record for record in records}
    return {
        str(epoch): (
            None
            if event is None or epoch < int(event["epoch"])
            else bool(predicate(by_epoch[epoch]))
        )
        for epoch in PERSISTENCE_CHECKPOINTS
    }


def _prefix_projection(projection: dict[str, Any]) -> dict[str, Any]:
    checkpoint = next(item for item in projection["checkpoints"] if item["epoch"] == 3)
    return {
        "initialization_canonical_sha256": projection["initialization_canonical_sha256"],
        "trained_canonical_sha256": checkpoint["trained_canonical_sha256"],
        "optimizer_canonical_sha256": checkpoint["optimizer_canonical_sha256"],
        "updates": projection["updates"][:9],
    }


def reconstruct_reference(
    rows: list[dict[str, Any]], *, seed: int, dataset_hash: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the accepted canonical-order reference without clipping-producer helpers."""
    configure_canonical_cpu_runtime(seed)
    canonical_rows = sorted(rows, key=lambda row: row["task_id"])
    if tuple(row["task_id"] for row in canonical_rows) != TASKS:
        raise ValueError("A2_CLIP_REFERENCE_TASK_ORDER")

    model = LockedPlanner(seed, "A2").cpu()
    initialization = canonical_state_dict_sha256(model.state_dict())
    optimizer_named = _optimizer_named_parameters(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in optimizer_named],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    updates: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []

    for epoch_index in range(MAX_EPOCH):
        for row in canonical_rows:
            action, arg1, arg2 = labels(row)
            valid = len(row["oracle_work_plan"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(canonical_task_encoding(row), action, arg1, arg2)
            flat = action[:, :valid].flatten()
            action_loss = F.cross_entropy(logits.action[:, :valid].flatten(0, 1), flat)
            operator_loss = float(action_loss.detach())
            arg1_loss = torch.zeros((), dtype=action_loss.dtype)
            arg2_loss = torch.zeros((), dtype=action_loss.dtype)
            loss = action_loss
            one = flat != ACTIONS["END"]
            two = (flat == ACTIONS["UNSTACK"]) | (flat == ACTIONS["STACK"])
            if one.any():
                arg1_loss = F.cross_entropy(
                    logits.arg1[:, :valid].flatten(0, 1)[one],
                    arg1[:, :valid].flatten()[one],
                )
                loss += arg1_loss
            if two.any():
                arg2_loss = F.cross_entropy(
                    logits.arg2[:, :valid].flatten(0, 1)[two],
                    arg2[:, :valid].flatten()[two],
                )
                loss += arg2_loss
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0,
                )
            )
            optimizer.step()
            updates.append(
                {
                    "update_index": len(updates),
                    "epoch_index": epoch_index,
                    "task_id": row["task_id"],
                    "operator_loss": operator_loss,
                    "arg1_pointer_loss": float(arg1_loss.detach()) if one.any() else None,
                    "arg2_pointer_loss": float(arg2_loss.detach()) if two.any() else None,
                    "total_loss": float(loss.detach()),
                    "operator_target_count": int(flat.numel()),
                    "arg1_target_count": int(one.sum()),
                    "arg2_target_count": int(two.sum()),
                    "gradient_norm": gradient_norm,
                    "gradient_clip_norm": 1.0,
                    "clipping_occurred": gradient_norm > 1.0,
                    "operator_position_weight": 1.0 / int(flat.numel()),
                }
            )
        epoch = epoch_index + 1
        epochs.append(_epoch_evidence(model, canonical_rows, seed=seed, epoch=epoch))
        if epoch in CHECKPOINTS:
            checkpoints.append(
                _checkpoint(model, optimizer, canonical_rows, seed=seed, epoch=epoch)
            )

    position0_event = _first(epochs, _position0_rescued)
    free_event = _first(epochs, _free_rescued)
    projection = {
        "seed": seed,
        "initialization_canonical_sha256": initialization,
        "updates": [{field: update[field] for field in REFERENCE_FIELDS} for update in updates],
        "checkpoints": checkpoints,
        "final_trained_canonical_sha256": canonical_state_dict_sha256(model.state_dict()),
        "final_optimizer_canonical_sha256": canonical_torch_object_sha256(optimizer.state_dict()),
        "rescue_events": {
            "first_position0_operator_rescue": position0_event,
            "first_full_free_running_rescue": free_event,
        },
        "rescue_persistence": {
            "position0_operator_rescue": _persistence(epochs, position0_event, _position0_rescued),
            "full_free_running_rescue": _persistence(epochs, free_event, _free_rescued),
        },
    }
    frozen_prefix = _frozen_control_projection(
        canonical_rows, seed=seed, dataset_hash=dataset_hash
    )
    if _prefix_projection(projection) != frozen_prefix:
        raise RuntimeError(f"A2_CLIP_REFERENCE_FROZEN_PREFIX:{seed}")
    return projection, frozen_prefix
