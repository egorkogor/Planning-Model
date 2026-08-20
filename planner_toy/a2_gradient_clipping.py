"""Train-only A2 gradient-clipping causal discrimination experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .a2_optimization_budget_trajectory import (
    PREFIX_TRACE_FIELDS,
    _checkpoint_evidence,
    _control_training,
    _train_rows,
)
from .a2_optimization_budget_trajectory import (
    SOURCE_FILES as BUDGET_SOURCE_FILES,
)
from .a2_sufficient_budget_task_order import (
    ARMS as REFERENCE_ORDER_ARMS,
)
from .a2_sufficient_budget_task_order import (
    _free_rescued,
    _persistence,
    _position0_rescued,
)
from .a2_sufficient_budget_task_order import (
    _train_arm as _reference_train_arm,
)
from .canonical import canonical_bytes, sha256
from .canonical_runtime import canonical_cpu_runtime_fingerprint, configure_canonical_cpu_runtime
from .learnability import (
    FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS,
    SEEDS,
    _read_only_diagnostic_pass,
    free_running_task,
    teacher_forced_task,
)
from .model import LockedPlanner, canonical_task_encoding
from .numeric_identity import canonical_state_dict_sha256, canonical_torch_object_sha256
from .quality import _optimizer_named_parameters
from .training import ACTIONS, labels

VERSION = "development-a2-gradient-clipping/0.1"
STATUS = "development-only-scientific-microexperiment"
VARIANT = "A2"
TASK_ID = "a2-gradient-clipping-v1"
EXPECTED_TRAIN_TASK_IDS = ("bw-00000001", "bw-00000002", "bw-00000003")
NONTRIVIAL_TASK_IDS = ("bw-00000002", "bw-00000003")
CANONICAL_ORDER = EXPECTED_TRAIN_TASK_IDS
ARMS: dict[str, float | None] = {"clip_1_0": 1.0, "clip_5_0": 5.0, "no_clip": None}
CHECKPOINT_EPOCHS = (3, 10, 30, 100)
PERSISTENCE_CHECKPOINT_EPOCHS = (10, 30, 100)
MAX_EPOCH = 100
UPDATES_PER_EPOCH = 3
EXPECTED_UPDATES = MAX_EPOCH * UPDATES_PER_EPOCH
OUTPUT_JSON = "a2-gradient-clipping.json"
OUTPUT_MARKDOWN = "A2_GRADIENT_CLIPPING_CAUSAL.md"
INTERPRETATION_LABEL = "SUPPORTED HYPOTHESIS / NOT PROVEN"
GRADIENT_HASH_VERSION = "a2-named-gradients-exact/1.0"
GRADIENT_EVIDENCE_COMMITMENT_VERSION = "a2-gradient-evidence-commitment/1.0"
INACTIVE_GRADIENT_MARKER = b"NO_GRAD"
ACTIVE_GRADIENT_MARKER = b"GRAD"
ROOT = Path(__file__).parents[1]
SOURCE_FILES = tuple(
    sorted(
        set(BUDGET_SOURCE_FILES)
        | {
            ".github/workflows/reviewer-execution-bridge.yml",
            "configs/fixed-cpu-target-1.0.json",
            "docs/evaluations/A2_GRADIENT_CLIPPING_CAUSAL_SPEC_RU.md",
            "planner_toy/a2_gradient_clipping.py",
            "planner_toy/a2_gradient_clipping_reference.py",
            "planner_toy/a2_gradient_clipping_validator.py",
            "planner_toy/a2_sufficient_budget_task_order.py",
            "planner_toy/a2_sufficient_budget_task_order_validator.py",
            "scripts/run_a2_gradient_clipping.py",
            "scripts/run_reviewer_execution_bridge.py",
        }
    )
)

OPTIMIZER_CONTRACT = {
    "name": "AdamW",
    "learning_rate": 3e-4,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "weight_decay": 0.01,
    "parameter_order": "planner_toy.quality._optimizer_named_parameters",
}

CLIPPING_CONTRACT = {
    "only_intended_causal_intervention": "gradient clipping policy",
    "clip_1_0": {"primitive": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0},
    "clip_5_0": {"primitive": "torch.nn.utils.clip_grad_norm_", "max_norm": 5.0},
    "no_clip": {"primitive": None, "max_norm": None},
    "gradient_hash_version": GRADIENT_HASH_VERSION,
    "gradient_hash_semantics": "name + dtype + shape + exact contiguous CPU bytes",
    "gradient_evidence_commitment_version": GRADIENT_EVIDENCE_COMMITMENT_VERSION,
    "common_pre_intervention_norm_field": "pre_intervention_global_l2_norm",
    "common_pre_intervention_norm_semantics": (
        "read-only global L2 over the same ordered active named gradients before intervention"
    ),
    "legacy_reference_norm_field": "gradient_norm",
    "legacy_reference_norm_semantics": (
        "clip primitive return for clipped arms; retained only for accepted reference equivalence"
    ),
    "threshold_predicate_field": "threshold_exceeded",
    "actual_mutation_field": "gradient_mutated",
    "actual_intervention_field": "intervention_applied",
    "actual_intervention_semantics": (
        "clip arm and gradient_before_sha256 != gradient_after_sha256"
    ),
}


def _git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def _validate_commit(commit: str) -> None:
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("A2_CLIP_IMPLEMENTATION_COMMIT_FORMAT")
    if _git_bytes("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise ValueError("A2_CLIP_IMPLEMENTATION_COMMIT_NOT_FOUND")
    if _git_bytes("merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise ValueError("A2_CLIP_IMPLEMENTATION_COMMIT_NOT_ANCESTOR")
    for path in SOURCE_FILES:
        if _git_bytes("show", f"{commit}:{path}").returncode:
            raise ValueError(f"A2_CLIP_IMPLEMENTATION_SOURCE_MISSING:{path}")


def source_identity_at_commit(commit: str) -> dict[str, Any]:
    _validate_commit(commit)
    files = []
    for path in SOURCE_FILES:
        result = _git_bytes("show", f"{commit}:{path}")
        files.append(
            {"path": path, "sha256": "sha256:" + hashlib.sha256(result.stdout).hexdigest()}
        )
    return {"source_files": files, "source_sha256": sha256(files)}


def _fixed_target_contract_identity() -> dict[str, Any]:
    contract = json.loads((ROOT / "configs/fixed-cpu-target-1.0.json").read_text(encoding="utf-8"))
    fields = (
        "target_contract_version",
        "target_name",
        "runner_image",
        "python_implementation",
        "python_version",
        "torch_version",
        "ATEN_CPU_CAPABILITY",
        "actual_atten_cpu_capability",
        "MKL_CBWR",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "torch_num_threads",
        "torch_num_interop_threads",
        "mkldnn_enabled",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "optimizer_foreach",
        "optimizer_fused",
    )
    return {"fields": {field: contract[field] for field in fields}, "sha256": sha256(contract)}


def _runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "canonical_cpu_runtime": canonical_cpu_runtime_fingerprint(),
        "fixed_target_contract": _fixed_target_contract_identity(),
    }


def _gradient_items(
    optimizer_named: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor | None]]:
    items: list[tuple[str, torch.Tensor | None]] = []
    for name, parameter in optimizer_named:
        if parameter.grad is None:
            items.append((name, None))
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"A2_CLIP_NONFINITE_GRADIENT:{name}")
        items.append((name, gradient))
    return items


def _gradient_parameter_manifest(
    optimizer_named: list[tuple[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "name": name,
            "dtype": str(parameter.dtype),
            "shape": list(parameter.shape),
        }
        for index, (name, parameter) in enumerate(optimizer_named)
    ]


def _named_gradient_sha256(items: list[tuple[str, torch.Tensor | None]]) -> str:
    digest = hashlib.sha256()
    digest.update(GRADIENT_HASH_VERSION.encode("ascii") + b"\0")
    for name, gradient in items:
        name_b = name.encode("utf-8")
        digest.update(len(name_b).to_bytes(8, "big") + name_b)
        if gradient is None:
            digest.update(len(INACTIVE_GRADIENT_MARKER).to_bytes(8, "big"))
            digest.update(INACTIVE_GRADIENT_MARKER)
            continue
        digest.update(len(ACTIVE_GRADIENT_MARKER).to_bytes(8, "big"))
        digest.update(ACTIVE_GRADIENT_MARKER)
        tensor = gradient.detach().cpu().contiguous()
        dtype_b = str(tensor.dtype).encode("ascii")
        digest.update(len(dtype_b).to_bytes(8, "big") + dtype_b)
        digest.update(tensor.ndim.to_bytes(8, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        data = tensor.numpy().tobytes()
        digest.update(len(data).to_bytes(8, "big") + data)
    return "sha256:" + digest.hexdigest()


def _global_l2_norm(items: list[tuple[str, torch.Tensor | None]]) -> float:
    norms = [
        torch.linalg.vector_norm(gradient, ord=2)
        for _, gradient in items
        if gradient is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), ord=2))


def _epoch_evidence(
    model: LockedPlanner,
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


def _first_event(records: list[dict[str, Any]], predicate) -> dict[str, int] | None:
    for record in records:
        if predicate(record):
            return {"epoch": int(record["epoch"]), "update_count": int(record["update_count"])}
    return None


def _train_arm(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    arm: str,
    max_epoch: int = MAX_EPOCH,
    checkpoint_epochs: tuple[int, ...] = CHECKPOINT_EPOCHS,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"A2_CLIP_ARM_UNKNOWN:{arm}")
    configure_canonical_cpu_runtime(seed)
    canonical_rows = sorted(rows, key=lambda row: row["task_id"])
    if tuple(row["task_id"] for row in canonical_rows) != CANONICAL_ORDER:
        raise ValueError("A2_CLIP_CANONICAL_TASK_ORDER")
    model = LockedPlanner(seed, VARIANT).cpu()
    initialization = canonical_state_dict_sha256(model.state_dict())
    optimizer_named = _optimizer_named_parameters(model)
    gradient_parameter_manifest = _gradient_parameter_manifest(optimizer_named)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in optimizer_named],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    threshold = ARMS[arm]
    updates: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []

    for epoch_index in range(max_epoch):
        for row in canonical_rows:
            action, arg1, arg2 = labels(row)
            valid = len(row["oracle_work_plan"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(canonical_task_encoding(row), action, arg1, arg2)
            flat = action[:, :valid].flatten()
            action_loss = F.cross_entropy(logits.action[:, :valid].flatten(0, 1), flat)
            operator_loss_value = float(action_loss.detach())
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

            before_items = _gradient_items(optimizer_named)
            before_hash = _named_gradient_sha256(before_items)
            common_pre_norm = _global_l2_norm(before_items)
            primitive_return_norm: float | None = None
            if threshold is None:
                legacy_pre_norm = common_pre_norm
            else:
                primitive_return_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for _, parameter in optimizer_named], threshold
                    )
                )
                legacy_pre_norm = primitive_return_norm

            after_items = _gradient_items(optimizer_named)
            post_norm = _global_l2_norm(after_items)
            after_hash = _named_gradient_sha256(after_items)
            threshold_exceeded = threshold is not None and common_pre_norm > threshold
            gradient_mutated = before_hash != after_hash
            intervention_applied = threshold is not None and gradient_mutated
            legacy_clipping_occurred = threshold is not None and legacy_pre_norm > threshold

            if threshold is None and gradient_mutated:
                raise RuntimeError(f"A2_CLIP_NO_CLIP_GRADIENT_MUTATION:{seed}:{len(updates)}")
            if intervention_applied and post_norm > threshold * (1.0 + 1e-5):
                raise RuntimeError(f"A2_CLIP_POST_INTERVENTION_NORM:{arm}:{seed}:{len(updates)}")

            optimizer.step()
            updates.append(
                {
                    "update_index": len(updates),
                    "epoch_index": epoch_index,
                    "task_id": row["task_id"],
                    "operator_loss": operator_loss_value,
                    "arg1_pointer_loss": float(arg1_loss.detach()) if one.any() else None,
                    "arg2_pointer_loss": float(arg2_loss.detach()) if two.any() else None,
                    "total_loss": float(loss.detach()),
                    "operator_target_count": int(flat.numel()),
                    "arg1_target_count": int(one.sum()),
                    "arg2_target_count": int(two.sum()),
                    "gradient_norm": legacy_pre_norm,
                    "gradient_clip_norm": threshold,
                    "clipping_policy": arm,
                    "clip_threshold": threshold,
                    "clipping_occurred": legacy_clipping_occurred,
                    "pre_intervention_global_l2_norm": common_pre_norm,
                    "clip_primitive_return_norm": primitive_return_norm,
                    "threshold_exceeded": threshold_exceeded,
                    "gradient_mutated": gradient_mutated,
                    "intervention_applied": intervention_applied,
                    "post_clip_global_l2_norm": post_norm,
                    "gradient_before_sha256": before_hash,
                    "gradient_after_sha256": after_hash,
                    "gradient_hash_version": GRADIENT_HASH_VERSION,
                    "operator_position_weight": 1.0 / int(flat.numel()),
                }
            )
        epoch = epoch_index + 1
        epochs.append(_epoch_evidence(model, canonical_rows, seed=seed, epoch=epoch))
        if epoch in checkpoint_epochs:
            checkpoints.append(
                _checkpoint_evidence(model, optimizer, canonical_rows, seed=seed, epoch=epoch)
            )

    position0_event = _first_event(epochs, _position0_rescued)
    free_event = _first_event(epochs, _free_rescued)
    return {
        "arm": arm,
        "clipping_policy": arm,
        "clip_threshold": threshold,
        "task_order": list(CANONICAL_ORDER),
        "seed": seed,
        "initialization_canonical_sha256": initialization,
        "final_trained_canonical_sha256": canonical_state_dict_sha256(model.state_dict()),
        "final_optimizer_canonical_sha256": canonical_torch_object_sha256(optimizer.state_dict()),
        "gradient_parameter_manifest": gradient_parameter_manifest,
        "gradient_parameter_manifest_sha256": sha256(gradient_parameter_manifest),
        "updates": updates,
        "checkpoints": checkpoints,
        "epoch_evidence": epochs,
        "rescue_events": {
            "first_position0_operator_rescue": position0_event,
            "first_full_free_running_rescue": free_event,
        },
        "rescue_persistence": {
            "position0_operator_rescue": _persistence(epochs, position0_event, _position0_rescued),
            "full_free_running_rescue": _persistence(epochs, free_event, _free_rescued),
        },
    }


def _reference_projection(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(result["seed"]),
        "initialization_canonical_sha256": result["initialization_canonical_sha256"],
        "updates": [
            {field: update[field] for field in PREFIX_TRACE_FIELDS}
            for update in result["updates"]
        ],
        "checkpoints": result["checkpoints"],
        "final_trained_canonical_sha256": result["final_trained_canonical_sha256"],
        "final_optimizer_canonical_sha256": result["final_optimizer_canonical_sha256"],
        "rescue_events": result["rescue_events"],
        "rescue_persistence": result["rescue_persistence"],
    }


def _candidate_control_projection(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized["updates"] = [
        {**update, "gradient_clip_norm": update["clip_threshold"]}
        for update in result["updates"]
    ]
    return _reference_projection(normalized)


def _control_equivalence(
    reference: dict[str, Any], candidate: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    left = _reference_projection(reference)
    right = _candidate_control_projection(candidate)
    if left != right:
        raise RuntimeError(f"A2_CLIP_CONTROL_WHOLE_TRAJECTORY_EQUIVALENCE:{seed}")
    left_hash = sha256(left)
    right_hash = sha256(right)
    return {
        "seed": seed,
        "status": "PASS",
        "scope": "WHOLE_300_UPDATE_TRAJECTORY",
        "trace_fields": list(PREFIX_TRACE_FIELDS),
        "reference_projection": left,
        "reference_projection_sha256": left_hash,
        "candidate_projection_sha256": right_hash,
        "prefix_9_update_projection_sha256": sha256(_control_prefix_projection(left)),
        "reference_historical_prefix_equivalence": reference.get("prefix_equivalence"),
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "reference_checkpoints": left["checkpoints"],
        "candidate_checkpoints": right["checkpoints"],
        "reference_final_trained_canonical_sha256": left["final_trained_canonical_sha256"],
        "candidate_final_trained_canonical_sha256": right["final_trained_canonical_sha256"],
        "reference_final_optimizer_canonical_sha256": left["final_optimizer_canonical_sha256"],
        "candidate_final_optimizer_canonical_sha256": right["final_optimizer_canonical_sha256"],
        "reference_rescue_events": left["rescue_events"],
        "candidate_rescue_events": right["rescue_events"],
        "reference_rescue_persistence": left["rescue_persistence"],
        "candidate_rescue_persistence": right["rescue_persistence"],
    }


def _control_prefix_projection(projection: dict[str, Any]) -> dict[str, Any]:
    checkpoint = next(item for item in projection["checkpoints"] if item["epoch"] == 3)
    return {
        "seed": projection["seed"],
        "initialization_canonical_sha256": projection["initialization_canonical_sha256"],
        "updates": projection["updates"][:9],
        "checkpoint_epoch_3": checkpoint,
    }


def _gradient_evidence_projection(result: dict[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "update_index",
        "epoch_index",
        "task_id",
        "clipping_policy",
        "clip_threshold",
        "gradient_norm",
        "pre_intervention_global_l2_norm",
        "clip_primitive_return_norm",
        "threshold_exceeded",
        "gradient_mutated",
        "intervention_applied",
        "post_clip_global_l2_norm",
        "clipping_occurred",
        "gradient_before_sha256",
        "gradient_after_sha256",
        "gradient_hash_version",
    )
    return [{field: update[field] for field in fields} for update in result["updates"]]


def _gradient_evidence_commitment(result: dict[str, Any]) -> dict[str, Any]:
    projection = _gradient_evidence_projection(result)
    return {
        "version": GRADIENT_EVIDENCE_COMMITMENT_VERSION,
        "arm": result["arm"],
        "seed": result["seed"],
        "update_count": len(projection),
        "sha256": sha256(projection),
    }


def _window_metrics(updates: list[dict[str, Any]]) -> dict[str, Any]:
    norms = [float(update["pre_intervention_global_l2_norm"]) for update in updates]
    interventions = [bool(update["intervention_applied"]) for update in updates]
    effects = [
        max(
            0.0,
            float(update["pre_intervention_global_l2_norm"])
            - float(update["post_clip_global_l2_norm"]),
        )
        for update in updates
    ]
    return {
        "update_count": len(updates),
        "clipped_update_count": sum(interventions),
        "clipped_update_fraction": sum(interventions) / len(updates) if updates else None,
        "max_pre_clip_global_l2_norm": max(norms) if norms else None,
        "mean_pre_clip_global_l2_norm": statistics.fmean(norms) if norms else None,
        "median_pre_clip_global_l2_norm": statistics.median(norms) if norms else None,
        "mean_clipping_effect_l2": statistics.fmean(effects) if effects else None,
        "max_clipping_effect_l2": max(effects) if effects else None,
    }


def _clipping_summary(result: dict[str, Any]) -> dict[str, Any]:
    events = result["rescue_events"]
    p0 = events["first_position0_operator_rescue"]
    free = events["first_full_free_running_rescue"]

    def through(event: dict[str, int] | None) -> dict[str, Any]:
        if event is None:
            return {"censored_at_update": EXPECTED_UPDATES, "event": None, "metrics": None}
        end = int(event["update_count"])
        return {
            "censored_at_update": None,
            "event": event,
            "metrics": _window_metrics(result["updates"][:end]),
        }

    return {
        "arm": result["arm"],
        "seed": result["seed"],
        "first_9_updates": _window_metrics(result["updates"][:9]),
        "through_first_position0_rescue_observation": through(p0),
        "through_first_full_free_running_rescue_observation": through(free),
        "full_trajectory": _window_metrics(result["updates"]),
    }


def _cross_seed_clipping_aggregates(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ARMS:
        arm_results = sorted(
            (result for result in results if result["arm"] == arm),
            key=lambda result: int(result["seed"]),
        )
        if [int(result["seed"]) for result in arm_results] != list(SEEDS):
            raise ValueError(f"A2_CLIP_AGGREGATE_SEED_COVERAGE:{arm}")
        updates = [update for result in arm_results for update in result["updates"]]
        output[arm] = {
            "seed_count": len(arm_results),
            "total_update_count": len(updates),
            "full_trajectory": _window_metrics(updates),
            "clipped_update_count_by_seed": {
                str(result["seed"]): sum(
                    bool(update["intervention_applied"]) for update in result["updates"]
                )
                for result in arm_results
            },
        }
    return output


def _event_delta(other: dict[str, int] | None, control: dict[str, int] | None) -> int | None:
    if other is None or control is None:
        return None
    return int(other["update_count"]) - int(control["update_count"])


def _paired_contrasts(results: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {(result["arm"], int(result["seed"])): result for result in results}
    output: dict[str, Any] = {}
    for arm in ("clip_5_0", "no_clip"):
        by_seed = {}
        for seed in SEEDS:
            control = indexed[("clip_1_0", seed)]
            other = indexed[(arm, seed)]
            control_events = control["rescue_events"]
            other_events = other["rescue_events"]
            control_clipped = sum(
                bool(update["intervention_applied"]) for update in control["updates"]
            )
            other_clipped = sum(
                bool(update["intervention_applied"]) for update in other["updates"]
            )
            persistence = {}
            for key in ("position0_operator_rescue", "full_free_running_rescue"):
                persistence[key] = {
                    epoch: {
                        "control": control["rescue_persistence"][key][epoch],
                        "intervention": other["rescue_persistence"][key][epoch],
                    }
                    for epoch in map(str, PERSISTENCE_CHECKPOINT_EPOCHS)
                }
            by_seed[str(seed)] = {
                "delta_first_position0_rescue_update_vs_clip_1_0": _event_delta(
                    other_events["first_position0_operator_rescue"],
                    control_events["first_position0_operator_rescue"],
                ),
                "delta_first_full_free_running_rescue_update_vs_clip_1_0": _event_delta(
                    other_events["first_full_free_running_rescue"],
                    control_events["first_full_free_running_rescue"],
                ),
                "control_clipped_update_count": control_clipped,
                "intervention_clipped_update_count": other_clipped,
                "intervention_clipped_update_fraction": other_clipped / len(other["updates"]),
                "persistence": persistence,
                "gradient_norm_full_trajectory": _window_metrics(other["updates"]),
            }
        output[arm] = {"by_seed": by_seed}
    return output


def _pre_intervention_update_projection(update: dict[str, Any]) -> dict[str, Any]:
    return {
        key: update[key]
        for key in (
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
            "pre_intervention_global_l2_norm",
            "gradient_before_sha256",
            "gradient_hash_version",
            "operator_position_weight",
        )
    }


def _intervention_consistency(results: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {(result["arm"], int(result["seed"])): result for result in results}
    output = {}
    for seed in SEEDS:
        by_arm = {arm: indexed[(arm, seed)] for arm in ARMS}
        first_intervention = None
        identical_before_first_intervention = 0
        for index in range(EXPECTED_UPDATES):
            projections = [
                _pre_intervention_update_projection(by_arm[arm]["updates"][index])
                for arm in ARMS
            ]
            if any(projection != projections[0] for projection in projections[1:]):
                raise RuntimeError(f"A2_CLIP_PREINTERVENTION_DIVERGENCE:{seed}:{index}")
            identical_before_first_intervention += 1
            if any(by_arm[arm]["updates"][index]["intervention_applied"] for arm in ARMS):
                first_intervention = index
                break

        clip5 = by_arm["clip_5_0"]
        no_clip = by_arm["no_clip"]
        clip5_count = sum(bool(update["intervention_applied"]) for update in clip5["updates"])
        clip5_first_intervention = None
        clip5_noclip_identical = 0
        for index in range(EXPECTED_UPDATES):
            left = _pre_intervention_update_projection(clip5["updates"][index])
            right = _pre_intervention_update_projection(no_clip["updates"][index])
            if left != right:
                raise RuntimeError(f"A2_CLIP_CLIP5_NOCLIP_PREINTERVENTION:{seed}:{index}")
            clip5_noclip_identical += 1
            if clip5["updates"][index]["intervention_applied"]:
                clip5_first_intervention = index
                break

        no_effect_equivalence = None
        if clip5_count == 0:
            left = {
                "losses": [
                    (
                        update["operator_loss"],
                        update["arg1_pointer_loss"],
                        update["arg2_pointer_loss"],
                        update["total_loss"],
                        update["gradient_before_sha256"],
                    )
                    for update in clip5["updates"]
                ],
                "final_model": clip5["final_trained_canonical_sha256"],
                "final_optimizer": clip5["final_optimizer_canonical_sha256"],
                "rescue_events": clip5["rescue_events"],
            }
            right = {
                "losses": [
                    (
                        update["operator_loss"],
                        update["arg1_pointer_loss"],
                        update["arg2_pointer_loss"],
                        update["total_loss"],
                        update["gradient_before_sha256"],
                    )
                    for update in no_clip["updates"]
                ],
                "final_model": no_clip["final_trained_canonical_sha256"],
                "final_optimizer": no_clip["final_optimizer_canonical_sha256"],
                "rescue_events": no_clip["rescue_events"],
            }
            if left != right:
                raise RuntimeError(f"A2_CLIP_CLIP5_NOCLIP_NO_EFFECT_EQUIVALENCE:{seed}")
            no_effect_equivalence = {"status": "PASS", "projection_sha256": sha256(left)}
        output[str(seed)] = {
            "first_actual_or_pregradient_difference_update_index": first_intervention,
            "all_arm_identical_preintervention_update_count": identical_before_first_intervention,
            "clip_5_0_clipped_update_count": clip5_count,
            "clip_5_0_first_actual_intervention_update_index": clip5_first_intervention,
            "clip_5_0_vs_no_clip_identical_preintervention_update_count": clip5_noclip_identical,
            "clip_5_0_vs_no_clip_no_effect_equivalence": no_effect_equivalence,
        }
    return output


def _produce_payload(*, implementation_commit: str) -> dict[str, Any]:
    configure_canonical_cpu_runtime()
    dataset, rows = _train_rows()
    if tuple(row["task_id"] for row in rows) != CANONICAL_ORDER:
        raise ValueError("A2_CLIP_TRAIN_TASK_ORDER")
    if set(CANONICAL_ORDER) & set(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS):
        raise ValueError("A2_CLIP_HELDOUT_ACCESS")
    source = source_identity_at_commit(implementation_commit)
    results: list[dict[str, Any]] = []
    equivalence: dict[str, Any] = {}
    for seed in SEEDS:
        historical_control = _control_training(
            rows, seed=seed, dataset_hash=dataset["frozen_dataset_lineage_hash"]
        )
        reference = _reference_train_arm(
            rows,
            seed=seed,
            arm="canonical_order",
            order=REFERENCE_ORDER_ARMS["canonical_order"],
            control=historical_control,
        )
        seed_results = {arm: _train_arm(rows, seed=seed, arm=arm) for arm in ARMS}
        equivalence[str(seed)] = _control_equivalence(
            reference, seed_results["clip_1_0"], seed=seed
        )
        init = {result["initialization_canonical_sha256"] for result in seed_results.values()}
        if len(init) != 1:
            raise RuntimeError(f"A2_CLIP_INITIALIZATION_MISMATCH:{seed}")
        results.extend(seed_results.values())

    payload: dict[str, Any] = {
        "experiment_version": VERSION,
        "schema_version": VERSION,
        "status": STATUS,
        "task_id": TASK_ID,
        "implementation_commit": implementation_commit,
        **source,
        "runtime": _runtime(),
        "variant": VARIANT,
        "seeds": list(SEEDS),
        "arms": {name: {"clip_threshold": threshold} for name, threshold in ARMS.items()},
        "canonical_task_order": list(CANONICAL_ORDER),
        "max_epoch": MAX_EPOCH,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "optimizer_updates_per_seed_arm": EXPECTED_UPDATES,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "persistence_checkpoint_epochs": list(PERSISTENCE_CHECKPOINT_EPOCHS),
        "optimizer_contract": OPTIMIZER_CONTRACT,
        "clipping_contract": CLIPPING_CONTRACT,
        "control_equivalence": {
            "reference": "accepted canonical sufficient-budget A2 canonical_order path",
            "required_status": "PASS",
            "by_seed": equivalence,
        },
        "rescue_definitions": {
            "first_position0_operator_rescue": (
                "earliest completed epoch where both train tasks 02 and 03 decode gold "
                "UNSTACK at position 0"
            ),
            "first_full_free_running_rescue": (
                "earliest completed epoch where both initially-unsatisfied train tasks "
                "02 and 03 are solved by true free-running execution"
            ),
        },
        "heldout_accessed": False,
        "heldout_task_ids": list(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS),
        "dataset": {
            "schema_version": dataset["schema_version"],
            "frozen_dataset_lineage_hash": dataset["frozen_dataset_lineage_hash"],
            "evaluated_train_split_hash": dataset["evaluated_train_split_hash"],
            "dataset_lineage_order": dataset["train_task_ids"],
            "evaluated_task_ids": list(CANONICAL_ORDER),
        },
        "arm_seed_results": results,
        "gradient_evidence_commitments": [
            _gradient_evidence_commitment(result) for result in results
        ],
        "clipping_summaries": [_clipping_summary(result) for result in results],
        "cross_seed_clipping_aggregates": _cross_seed_clipping_aggregates(results),
        "paired_causal_contrasts": _paired_contrasts(results),
        "intervention_consistency": _intervention_consistency(results),
        "interpretation_policy": {
            "label": INTERPRETATION_LABEL,
            "producer_scientific_verdict": None,
            "validator_scientific_verdict": None,
            "reviewer_owns_interpretation": True,
            "p_value_testing": None,
            "scope": "A2 optimization dynamics only",
        },
        "go_latent": "NOT EVALUATED",
    }
    payload["canonical_identity"] = sha256(payload)
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# A2 gradient clipping causal discrimination",
        "",
        f"- Version: `{payload['experiment_version']}`",
        f"- Status: `{payload['status']}`",
        f"- Implementation: `{payload['implementation_commit']}`",
        f"- Source: `{payload['source_sha256']}`",
        f"- Arms: `{list(payload['arms'])}`",
        f"- Seeds: `{payload['seeds']}`",
        "- Budget: `100 epochs / 300 updates per seed-arm`",
        "- Held-out accessed: `false`",
        "- Control equivalence: `PASS required for all seeds`",
        "- Scientific verdict: `reviewer only; not emitted by producer`",
        "- GO_LATENT: `NOT EVALUATED`",
        "",
        "## Claim-bearing source",
        "",
        f"JSON evidence: `{OUTPUT_JSON}`. This Markdown is derivative only.",
        "",
    ]
    return "\n".join(lines)


def _validate_payload_invariants(payload: dict[str, Any]) -> None:
    if (
        payload.get("experiment_version") != VERSION
        or payload.get("schema_version") != VERSION
        or payload.get("status") != STATUS
    ):
        raise ValueError("A2_CLIP_VERSION_OR_STATUS")
    if payload.get("task_id") != TASK_ID or payload.get("variant") != VARIANT:
        raise ValueError("A2_CLIP_SCOPE")
    if payload.get("seeds") != list(SEEDS):
        raise ValueError("A2_CLIP_SEEDS")
    expected_arms = {name: {"clip_threshold": threshold} for name, threshold in ARMS.items()}
    if payload.get("arms") != expected_arms:
        raise ValueError("A2_CLIP_ARMS")
    if payload.get("canonical_task_order") != list(CANONICAL_ORDER):
        raise ValueError("A2_CLIP_TASK_ORDER")
    if (
        payload.get("max_epoch") != MAX_EPOCH
        or payload.get("optimizer_updates_per_seed_arm") != EXPECTED_UPDATES
    ):
        raise ValueError("A2_CLIP_BUDGET")
    if payload.get("heldout_accessed") is not False:
        raise ValueError("A2_CLIP_HELDOUT_ACCESS")
    if payload.get("heldout_task_ids") != list(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS):
        raise ValueError("A2_CLIP_HELDOUT_TASK_IDS")
    if payload.get("optimizer_contract") != OPTIMIZER_CONTRACT:
        raise ValueError("A2_CLIP_OPTIMIZER_CONTRACT")
    if payload.get("clipping_contract") != CLIPPING_CONTRACT:
        raise ValueError("A2_CLIP_CLIPPING_CONTRACT")
    if payload.get("go_latent") != "NOT EVALUATED":
        raise ValueError("A2_CLIP_GO_LATENT")
    identity = payload.get("canonical_identity")
    unsigned = {key: value for key, value in payload.items() if key != "canonical_identity"}
    if identity != sha256(unsigned):
        raise ValueError("A2_CLIP_CANONICAL_IDENTITY")


def run(output: Path, *, implementation_commit: str) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("A2_CLIP_OUTPUT_NOT_EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    payload = _produce_payload(implementation_commit=implementation_commit)
    _validate_payload_invariants(payload)
    from .a2_gradient_clipping_validator import validate_claims_from_evidence

    validate_claims_from_evidence(payload, implementation_commit=implementation_commit)
    (output / OUTPUT_JSON).write_bytes(canonical_bytes(payload) + b"\n")
    (output / OUTPUT_MARKDOWN).write_text(render_markdown(payload), encoding="utf-8")
    return payload


def validate_experiment(output: Path, *, implementation_commit: str) -> dict[str, Any]:
    expected = {OUTPUT_JSON, OUTPUT_MARKDOWN}
    if not output.is_dir() or {path.name for path in output.iterdir()} != expected:
        raise ValueError("A2_CLIP_OUTPUT_COVERAGE")
    payload = json.loads((output / OUTPUT_JSON).read_text(encoding="utf-8"))
    _validate_payload_invariants(payload)
    if payload.get("implementation_commit") != implementation_commit:
        raise ValueError("A2_CLIP_IMPLEMENTATION_COMMIT_MISMATCH")
    source = source_identity_at_commit(implementation_commit)
    if (
        payload.get("source_sha256") != source["source_sha256"]
        or payload.get("source_files") != source["source_files"]
    ):
        raise ValueError("A2_CLIP_SOURCE_IDENTITY_MISMATCH")
    if (output / OUTPUT_MARKDOWN).read_text(encoding="utf-8") != render_markdown(payload):
        raise ValueError("A2_CLIP_MARKDOWN_MISMATCH")
    from .a2_gradient_clipping_validator import validate_claims_from_evidence

    return validate_claims_from_evidence(payload, implementation_commit=implementation_commit)
