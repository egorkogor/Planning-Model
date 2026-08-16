"""Development-only train-only A2 optimization-budget trajectory.

The historical trainer/model/objective are not modified. This module mirrors the canonical
A2 update rule into one prefix-preserving 100-epoch trajectory and fails closed unless the
3-epoch prefix exactly reproduces the real frozen historical control.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .canonical import canonical_bytes, sha256
from .canonical_runtime import configure_canonical_cpu_runtime
from .learnability import (
    FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS,
    SEEDS,
    _read_only_diagnostic_pass,
    _train_a2_with_loss_trace,
    free_running_task,
    teacher_forced_task,
)
from .learnability import (
    SOURCE_FILES as LEARNABILITY_SOURCE_FILES,
)
from .model import LockedPlanner, canonical_task_encoding
from .numeric_identity import (
    canonical_state_dict_sha256,
    canonical_torch_object_sha256,
)
from .quality import _optimizer_named_parameters
from .train_only_dataset import (
    FROZEN_DATASET_LINEAGE_HASH_V1,
    generate_train_only,
)
from .training import ACTIONS, labels

VERSION = "development-a2-optimization-budget-trajectory/0.1"
STATUS = "development-only-scientific-microexperiment"
VARIANT = "A2"
EXPECTED_TRAIN_TASK_IDS = ("bw-00000001", "bw-00000002", "bw-00000003")
CANONICAL_ORDER = EXPECTED_TRAIN_TASK_IDS
CHECKPOINT_EPOCHS = (3, 10, 30, 100)
MAX_EPOCH = 100
OUTPUT_JSON = "a2-optimization-budget-trajectory.json"
OUTPUT_MARKDOWN = "A2_OPTIMIZATION_BUDGET_TRAJECTORY.md"
INTERPRETATION_LABEL = "SUPPORTED HYPOTHESIS / NOT PROVEN"
ROOT = Path(__file__).parents[1]
SOURCE_FILES = tuple(
    sorted(
        set(LEARNABILITY_SOURCE_FILES)
        | {
            ".github/workflows/a2-optimization-budget-trajectory.yml",
            "docs/evaluations/A2_OPTIMIZATION_BUDGET_TRAJECTORY_SPEC_RU.md",
            "planner_toy/a2_optimization_budget_trajectory.py",
            "planner_toy/a2_optimization_budget_trajectory_validator.py",
            "requirements.lock",
            "scripts/run_a2_optimization_budget_trajectory.py",
        }
    )
)

PREFIX_TRACE_FIELDS = (
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


def _git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def _validate_commit(commit: str) -> None:
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("A2_BUDGET_IMPLEMENTATION_COMMIT_FORMAT")
    if _git_bytes("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise ValueError("A2_BUDGET_IMPLEMENTATION_COMMIT_NOT_FOUND")
    for path in SOURCE_FILES:
        if _git_bytes("show", f"{commit}:{path}").returncode:
            raise ValueError(f"A2_BUDGET_IMPLEMENTATION_SOURCE_MISSING:{path}")


def source_identity_at_commit(commit: str) -> dict[str, Any]:
    _validate_commit(commit)
    files = []
    for path in SOURCE_FILES:
        result = _git_bytes("show", f"{commit}:{path}")
        files.append(
            {
                "path": path,
                "sha256": "sha256:" + hashlib.sha256(result.stdout).hexdigest(),
            }
        )
    return {"source_files": files, "source_sha256": sha256(files)}


def _runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _train_rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = generate_train_only()
    rows = list(dataset["train"])
    ids = {row["task_id"] for row in rows}
    if ids != set(EXPECTED_TRAIN_TASK_IDS) or len(rows) != len(EXPECTED_TRAIN_TASK_IDS):
        raise ValueError("A2_BUDGET_TRAIN_TASK_COVERAGE_MISMATCH")
    if ids & set(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS):
        raise ValueError("A2_BUDGET_HELDOUT_ACCESS")
    rows.sort(key=lambda row: row["task_id"])
    return dataset, rows


def _control_training(
    rows: list[dict[str, Any]], *, seed: int, dataset_hash: str
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="a2-budget-control-") as temp:
        _model, checkpoint, trace = _train_a2_with_loss_trace(
            rows, seed, Path(temp), dataset_hash
        )
    updates = []
    for update in trace:
        count = int(update["operator_target_count"])
        updates.append({**update, "operator_position_weight": 1.0 / count})
    return {
        "initialization_canonical_sha256": checkpoint[
            "canonical_initialization_state_dict_sha256"
        ],
        "trained_canonical_sha256": checkpoint["canonical_trained_state_dict_sha256"],
        "optimizer_canonical_sha256": checkpoint["canonical_optimizer_state_sha256"],
        "updates": updates,
    }


def _prefix_projection(training: dict[str, Any], *, limit: int = 9) -> dict[str, Any]:
    return {
        "initialization_canonical_sha256": training["initialization_canonical_sha256"],
        "trained_canonical_sha256": training["trained_canonical_sha256"],
        "optimizer_canonical_sha256": training["optimizer_canonical_sha256"],
        "updates": [
            {field: update[field] for field in PREFIX_TRACE_FIELDS}
            for update in training["updates"][:limit]
        ],
    }


def _assert_prefix_equivalence(
    control: dict[str, Any], prefix: dict[str, Any], *, seed: int
) -> None:
    if control["initialization_canonical_sha256"] != prefix[
        "initialization_canonical_sha256"
    ]:
        raise RuntimeError(f"A2_BUDGET_PREFIX_EQUIVALENCE_INITIALIZATION:{seed}")
    if control["trained_canonical_sha256"] != prefix["trained_canonical_sha256"]:
        raise RuntimeError(f"A2_BUDGET_PREFIX_EQUIVALENCE_TRAINED:{seed}")
    if control["optimizer_canonical_sha256"] != prefix["optimizer_canonical_sha256"]:
        raise RuntimeError(f"A2_BUDGET_PREFIX_EQUIVALENCE_OPTIMIZER:{seed}")
    left = _prefix_projection(control)["updates"]
    right = _prefix_projection(prefix)["updates"]
    if left != right:
        raise RuntimeError(f"A2_BUDGET_PREFIX_EQUIVALENCE_TRACE:{seed}")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _task_teacher_summary(task: dict[str, Any]) -> dict[str, Any]:
    positions = task["positions"]
    non_end = [row for row in positions if row["gold_operator"] != "END"]
    end = [row for row in positions if row["gold_operator"] == "END"]
    arg1 = [row for row in positions if row["has_arg1_target"]]
    arg2 = [row for row in positions if row["has_arg2_target"]]
    return {
        "operator_accuracy": _rate([row["operator_correct"] for row in positions]),
        "non_end_operator_accuracy": _rate([row["operator_correct"] for row in non_end]),
        "end_accuracy": _rate([row["operator_correct"] for row in end]),
        "arg1_accuracy": _rate([bool(row["arg1_correct"]) for row in arg1]),
        "arg2_accuracy": _rate([bool(row["arg2_correct"]) for row in arg2]),
        "joint_step_accuracy": _rate([row["joint_step_correct"] for row in positions]),
    }


def _teacher_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [position for task in tasks for position in task["positions"]]
    non_end = [row for row in positions if row["gold_operator"] != "END"]
    end = [row for row in positions if row["gold_operator"] == "END"]
    arg1 = [row for row in positions if row["has_arg1_target"]]
    arg2 = [row for row in positions if row["has_arg2_target"]]
    pos0_unstack = [
        row
        for row in positions
        if row["position_index"] == 0 and row["gold_operator"] == "UNSTACK"
    ]
    pos4_end = [
        row
        for row in positions
        if row["position_index"] == 4 and row["gold_operator"] == "END"
    ]
    position0_by_task = {}
    for task in tasks:
        row = task["positions"][0]
        position0_by_task[task["task_id"]] = {
            "gold_operator": row["gold_operator"],
            "predicted_operator": row["predicted_operator"],
            "operator_correct": row["operator_correct"],
            "probability_gold_operator": row["probability_gold_operator"],
            "operator_nll": row["operator_nll"],
            "probability_end": row["probability_end"],
        }
    end_probs = {
        task_id: float(record["probability_end"])
        for task_id, record in position0_by_task.items()
    }
    nontrivial_mean = (end_probs["bw-00000002"] + end_probs["bw-00000003"]) / 2
    return {
        "aggregate": {
            "operator_accuracy": _rate([row["operator_correct"] for row in positions]),
            "non_end_operator_accuracy": _rate(
                [row["operator_correct"] for row in non_end]
            ),
            "end_accuracy": _rate([row["operator_correct"] for row in end]),
            "arg1_accuracy": _rate([bool(row["arg1_correct"]) for row in arg1]),
            "arg2_accuracy": _rate([bool(row["arg2_correct"]) for row in arg2]),
            "joint_step_accuracy": _rate(
                [row["joint_step_correct"] for row in positions]
            ),
            "predicted_end_rate": _rate(
                [row["predicted_operator"] == "END" for row in positions]
            ),
        },
        "per_task": {task["task_id"]: _task_teacher_summary(task) for task in tasks},
        "position0_by_task": position0_by_task,
        "position0_unstack": {
            "target_count": len(pos0_unstack),
            "accuracy": _rate([row["operator_correct"] for row in pos0_unstack]),
            "mean_gold_operator_probability": _mean(
                [float(row["probability_gold_operator"]) for row in pos0_unstack]
            ),
            "mean_operator_nll": _mean(
                [float(row["operator_nll"]) for row in pos0_unstack]
            ),
            "mean_end_probability": _mean(
                [float(row["probability_end"]) for row in pos0_unstack]
            ),
        },
        "position4_end": {
            "target_count": len(pos4_end),
            "accuracy": _rate([row["operator_correct"] for row in pos4_end]),
            "mean_gold_operator_probability": _mean(
                [float(row["probability_gold_operator"]) for row in pos4_end]
            ),
            "mean_operator_nll": _mean(
                [float(row["operator_nll"]) for row in pos4_end]
            ),
            "mean_end_probability": _mean(
                [float(row["probability_end"]) for row in pos4_end]
            ),
        },
        "position0_task_discrimination": {
            "task01_end_probability": end_probs["bw-00000001"],
            "task02_end_probability": end_probs["bw-00000002"],
            "task03_end_probability": end_probs["bw-00000003"],
            "nontrivial_mean_end_probability": nontrivial_mean,
            "task01_minus_nontrivial_mean_end_probability": (
                end_probs["bw-00000001"] - nontrivial_mean
            ),
            "end_probability_range": max(end_probs.values()) - min(end_probs.values()),
        },
    }


def _free_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    unsatisfied = [task for task in tasks if not task["initial_goal_satisfied"]]
    return {
        "task_count": len(tasks),
        "exact_plan_rate": _rate([task["exact_plan_match"] for task in tasks]),
        "goal_success_rate": _rate([task["final_goal_success"] for task in tasks]),
        "initially_unsatisfied_goal_success_rate": _rate(
            [task["final_goal_success"] for task in unsatisfied]
        ),
        "zero_action_rate": _rate([task["predicted_plan_length"] == 0 for task in tasks]),
    }


def _position0_epoch_evidence(
    model: LockedPlanner, rows: list[dict[str, Any]], *, seed: int, epoch: int
) -> dict[str, Any]:
    with _read_only_diagnostic_pass(model):
        tasks = [
            teacher_forced_task(model, row, split="train", seed=seed) for row in rows
        ]
    evidence = []
    for task in tasks:
        position = task["positions"][0]
        evidence.append(
            {
                "task_id": task["task_id"],
                "gold_operator": position["gold_operator"],
                "predicted_operator": position["predicted_operator"],
                "operator_correct": position["operator_correct"],
                "probability_gold_operator": position["probability_gold_operator"],
                "operator_nll": position["operator_nll"],
                "probability_end": position["probability_end"],
            }
        )
    return {"epoch": epoch, "update_count": epoch * len(rows), "tasks": evidence}


def _checkpoint_evidence(
    model: LockedPlanner,
    optimizer: torch.optim.Optimizer,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    with _read_only_diagnostic_pass(model):
        teacher = [
            teacher_forced_task(model, row, split="train", seed=seed) for row in rows
        ]
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


def _first_rescue_from_epoch_evidence(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        rescued = [
            task["task_id"]
            for task in record["tasks"]
            if task["gold_operator"] == "UNSTACK" and task["operator_correct"]
        ]
        if rescued:
            return {
                "epoch": record["epoch"],
                "update_count": record["update_count"],
                "task_ids": rescued,
            }
    return None


def _train_trajectory(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    control: dict[str, Any],
    max_epoch: int = MAX_EPOCH,
    checkpoint_epochs: tuple[int, ...] = CHECKPOINT_EPOCHS,
) -> dict[str, Any]:
    configure_canonical_cpu_runtime(seed)
    model = LockedPlanner(seed, VARIANT).cpu()
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
    position0_epochs: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    prefix_equivalence = None
    for epoch_index in range(max_epoch):
        for row in rows:
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
                    "operator_loss": operator_loss_value,
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
        position0_epochs.append(
            _position0_epoch_evidence(model, rows, seed=seed, epoch=epoch)
        )
        if epoch in checkpoint_epochs:
            checkpoint = _checkpoint_evidence(
                model, optimizer, rows, seed=seed, epoch=epoch
            )
            checkpoints.append(checkpoint)
            if epoch == 3:
                prefix = {
                    "initialization_canonical_sha256": initialization,
                    "trained_canonical_sha256": checkpoint["trained_canonical_sha256"],
                    "optimizer_canonical_sha256": checkpoint["optimizer_canonical_sha256"],
                    "updates": updates[:9],
                }
                _assert_prefix_equivalence(control, prefix, seed=seed)
                prefix_equivalence = {
                    "seed": seed,
                    "status": "PASS",
                    "purpose": "NON_SCIENTIFIC_FROZEN_3_EPOCH_PREFIX_EQUIVALENCE",
                    "trace_fields": list(PREFIX_TRACE_FIELDS),
                    "control": _prefix_projection(control),
                    "trajectory_prefix": _prefix_projection(prefix),
                }
    if max_epoch >= 3 and prefix_equivalence is None:
        raise RuntimeError(f"A2_BUDGET_PREFIX_EQUIVALENCE_NOT_EVALUATED:{seed}")
    return {
        "seed": seed,
        "initialization_canonical_sha256": initialization,
        "final_trained_canonical_sha256": canonical_state_dict_sha256(model.state_dict()),
        "final_optimizer_canonical_sha256": canonical_torch_object_sha256(
            optimizer.state_dict()
        ),
        "updates": updates,
        "checkpoints": checkpoints,
        "position0_epoch_evidence": position0_epochs,
        "first_position0_unstack_rescue": _first_rescue_from_epoch_evidence(
            position0_epochs
        ),
        "prefix_equivalence": prefix_equivalence,
    }


def _aggregate_checkpoint(seed_results: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    teacher = []
    free = []
    discriminations = []
    for result in seed_results:
        checkpoint = next(item for item in result["checkpoints"] if item["epoch"] == epoch)
        teacher.extend(checkpoint["teacher_forced"])
        free.extend(checkpoint["free_running"])
        discriminations.append(
            checkpoint["teacher_forced_summary"]["position0_task_discrimination"]
        )
    summary = _teacher_summary(teacher)
    return {
        "epoch": epoch,
        "update_count_per_seed": epoch * len(EXPECTED_TRAIN_TASK_IDS),
        "teacher_forced": summary,
        "free_running": _free_summary(free),
        "mean_within_seed_position0_task_discrimination": {
            key: _mean([float(item[key]) for item in discriminations])
            for key in discriminations[0]
        },
    }


def _global_first_rescue(seed_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for result in seed_results:
        rescue = result["first_position0_unstack_rescue"]
        if rescue is not None:
            candidates.append({"seed": result["seed"], **rescue})
    if not candidates:
        return None
    first_epoch = min(item["epoch"] for item in candidates)
    return {
        "epoch": first_epoch,
        "events": [item for item in candidates if item["epoch"] == first_epoch],
    }


def _produce_payload(*, implementation_commit: str) -> dict[str, Any]:
    configure_canonical_cpu_runtime()
    dataset, rows = _train_rows()
    source = source_identity_at_commit(implementation_commit)
    results = []
    for seed in SEEDS:
        control = _control_training(
            rows,
            seed=seed,
            dataset_hash=dataset["frozen_dataset_lineage_hash"],
        )
        results.append(_train_trajectory(rows, seed=seed, control=control))
    payload: dict[str, Any] = {
        "schema_version": VERSION,
        "status": STATUS,
        "implementation_commit": implementation_commit,
        **source,
        "runtime": _runtime(),
        "variant": VARIANT,
        "seeds": list(SEEDS),
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "max_epoch": MAX_EPOCH,
        "canonical_task_order": list(CANONICAL_ORDER),
        "canonical_objective": "per-task-mean operator/arg1/arg2 cross-entropy",
        "canonical_optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
        },
        "hypothesis": {
            "label": INTERPRETATION_LABEL,
            "question": (
                "Is the canonical 9-update/3-epoch budget itself sufficient for basic A2 "
                "memorization and position-0 task discrimination?"
            ),
            "causal_result": None,
        },
        "heldout_accessed": False,
        "heldout_task_ids": list(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS),
        "training_policy_changed": True,
        "frozen_science_changed": False,
        "go_latent": "NOT EVALUATED",
        "dataset": {
            "schema_version": dataset["schema_version"],
            "frozen_dataset_lineage_hash": FROZEN_DATASET_LINEAGE_HASH_V1,
            "evaluated_train_split_hash": dataset["evaluated_train_split_hash"],
            "dataset_lineage_order": dataset["train_task_ids"],
            "evaluated_task_ids": [row["task_id"] for row in rows],
        },
        "seed_results": results,
        "checkpoint_aggregates": {
            str(epoch): _aggregate_checkpoint(results, epoch) for epoch in CHECKPOINT_EPOCHS
        },
        "trajectory_claims": {
            "first_position0_unstack_rescue_global": _global_first_rescue(results),
            "first_position0_unstack_rescue_by_seed": {
                str(result["seed"]): result["first_position0_unstack_rescue"]
                for result in results
            },
        },
        "interpretation_policy": {
            "label": INTERPRETATION_LABEL,
            "automatic_gate": None,
            "scientific_status": "REDESIGN",
            "rescue_supports": "INSUFFICIENT_OPTIMIZATION_BUDGET_OR_UNDER_TRAINING",
            "persistent_failure_weakens": "BUDGET_ONLY_EXPLANATION",
        },
    }
    payload["canonical_identity"] = sha256(payload)
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# A2 optimization-budget trajectory",
        "",
        f"- Version: `{payload['schema_version']}`",
        f"- Implementation: `{payload['implementation_commit']}`",
        f"- Source: `{payload['source_sha256']}`",
        f"- Seeds: `{payload['seeds']}`",
        f"- Checkpoints: `{payload['checkpoint_epochs']}`",
        "- Frozen 3-epoch prefix equivalence: `PASS` for every seed",
        "- Held-out accessed: `false`",
        "- GO_LATENT: `NOT EVALUATED`",
        "",
        "## Checkpoints",
        "",
    ]
    for epoch in CHECKPOINT_EPOCHS:
        checkpoint = payload["checkpoint_aggregates"][str(epoch)]
        teacher = checkpoint["teacher_forced"]
        free = checkpoint["free_running"]
        lines.extend(
            [
                f"### Epoch {epoch}",
                f"- operator accuracy: `{teacher['aggregate']['operator_accuracy']}`",
                "- position-0 UNSTACK accuracy: "
                f"`{teacher['position0_unstack']['accuracy']}`",
                "- position-0 UNSTACK mean END probability: "
                f"`{teacher['position0_unstack']['mean_end_probability']}`",
                f"- free-running goal success: `{free['goal_success_rate']}`",
                "- initially-unsatisfied goal success: "
                f"`{free['initially_unsatisfied_goal_success_rate']}`",
                f"- zero-action rate: `{free['zero_action_rate']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            f"`{INTERPRETATION_LABEL}`",
            "",
            "This artifact has no automatic scientific gate. More-update rescue can support",
            "under-training as a major cause; persistent failure can weaken budget-only",
            "explanations. Neither outcome licenses A3/latent/semantic-target conclusions.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_payload_invariants(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != VERSION or payload.get("status") != STATUS:
        raise ValueError("A2_BUDGET_VERSION_OR_STATUS")
    if payload.get("variant") != VARIANT or payload.get("seeds") != list(SEEDS):
        raise ValueError("A2_BUDGET_SCOPE")
    if payload.get("checkpoint_epochs") != list(CHECKPOINT_EPOCHS):
        raise ValueError("A2_BUDGET_CHECKPOINTS")
    if payload.get("max_epoch") != MAX_EPOCH:
        raise ValueError("A2_BUDGET_MAX_EPOCH")
    if payload.get("canonical_task_order") != list(CANONICAL_ORDER):
        raise ValueError("A2_BUDGET_TASK_ORDER")
    if payload.get("heldout_accessed") is not False:
        raise ValueError("A2_BUDGET_HELDOUT_ACCESS")
    if payload.get("frozen_science_changed") is not False:
        raise ValueError("A2_BUDGET_FROZEN_SCIENCE_FLAG")
    if payload.get("go_latent") != "NOT EVALUATED":
        raise ValueError("A2_BUDGET_GO_LATENT_SCOPE")
    if payload.get("dataset", {}).get("evaluated_task_ids") != list(
        EXPECTED_TRAIN_TASK_IDS
    ):
        raise ValueError("A2_BUDGET_TRAIN_TASK_COVERAGE")
    identity = payload.get("canonical_identity")
    unsigned = {key: value for key, value in payload.items() if key != "canonical_identity"}
    if identity != sha256(unsigned):
        raise ValueError("A2_BUDGET_CANONICAL_IDENTITY")


def run(output: Path, *, implementation_commit: str) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("A2_BUDGET_OUTPUT_NOT_EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    payload = _produce_payload(implementation_commit=implementation_commit)
    _validate_payload_invariants(payload)
    from .a2_optimization_budget_trajectory_validator import validate_claims_from_evidence

    validate_claims_from_evidence(payload)
    (output / OUTPUT_JSON).write_bytes(canonical_bytes(payload) + b"\n")
    (output / OUTPUT_MARKDOWN).write_text(render_markdown(payload), encoding="utf-8")
    return payload


def validate_trajectory(output: Path, *, implementation_commit: str) -> dict[str, Any]:
    expected_names = {OUTPUT_JSON, OUTPUT_MARKDOWN}
    if not output.is_dir() or {path.name for path in output.iterdir()} != expected_names:
        raise ValueError("A2_BUDGET_OUTPUT_COVERAGE")
    payload = json.loads((output / OUTPUT_JSON).read_text(encoding="utf-8"))
    _validate_payload_invariants(payload)
    if payload.get("implementation_commit") != implementation_commit:
        raise ValueError("A2_BUDGET_IMPLEMENTATION_COMMIT_MISMATCH")
    expected_source = source_identity_at_commit(implementation_commit)
    if payload.get("source_sha256") != expected_source["source_sha256"]:
        raise ValueError("A2_BUDGET_SOURCE_IDENTITY_MISMATCH")
    if payload.get("source_files") != expected_source["source_files"]:
        raise ValueError("A2_BUDGET_SOURCE_FILES_MISMATCH")
    if (output / OUTPUT_MARKDOWN).read_text(encoding="utf-8") != render_markdown(payload):
        raise ValueError("A2_BUDGET_MARKDOWN_MISMATCH")
    from .a2_optimization_budget_trajectory_validator import validate_claims_from_evidence

    validation = validate_claims_from_evidence(payload)
    return {
        "valid": True,
        "canonical_identity": payload["canonical_identity"],
        "source_sha256": payload["source_sha256"],
        "heldout_accessed": False,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "seeds": list(SEEDS),
        "automatic_gate": None,
        **validation,
    }
