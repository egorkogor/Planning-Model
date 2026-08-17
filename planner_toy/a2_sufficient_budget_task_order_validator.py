"""Independent persisted-evidence validator for A2 sufficient-budget task-order experiment."""

from __future__ import annotations

import hashlib
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from .a2_optimization_budget_trajectory import PREFIX_TRACE_FIELDS
from .a2_optimization_budget_trajectory import SOURCE_FILES as BUDGET_SOURCE_FILES
from .canonical import sha256
from .canonical_runtime import configure_canonical_cpu_runtime
from .dataset import task_from_row
from .domain import apply_action, goal_satisfied, validate_state
from .e2e import parse_nonterminal_step
from .learnability import _train_a2_with_loss_trace
from .train_only_dataset import generate_train_only

SEEDS = (17, 29, 43)
TASKS = ("bw-00000001", "bw-00000002", "bw-00000003")
NONTRIVIAL = ("bw-00000002", "bw-00000003")
ARMS = {
    "canonical_order": ("bw-00000001", "bw-00000002", "bw-00000003"),
    "task01_middle": ("bw-00000002", "bw-00000001", "bw-00000003"),
    "task01_last": ("bw-00000002", "bw-00000003", "bw-00000001"),
}
CHECKPOINTS = (3, 10, 30, 100)
PERSISTENCE_CHECKPOINTS = (10, 30, 100)
MAX_EPOCH = 100
ROOT = Path(__file__).parents[1]
SOURCE_FILES = tuple(
    sorted(
        set(BUDGET_SOURCE_FILES)
        | {
            ".github/workflows/a2-sufficient-budget-task-order.yml",
            "docs/evaluations/A2_SUFFICIENT_BUDGET_TASK_ORDER_SPEC_RU.md",
            "planner_toy/a2_sufficient_budget_task_order.py",
            "planner_toy/a2_sufficient_budget_task_order_validator.py",
            "scripts/run_a2_sufficient_budget_task_order.py",
        }
    )
)


def _git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def _source_identity_at_commit(commit: str) -> dict[str, Any]:
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("A2_ORDER_VALIDATOR_IMPLEMENTATION_FORMAT")
    if _git_bytes("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise ValueError("A2_ORDER_VALIDATOR_IMPLEMENTATION_NOT_FOUND")
    files = []
    for path in SOURCE_FILES:
        result = _git_bytes("show", f"{commit}:{path}")
        if result.returncode:
            raise ValueError(f"A2_ORDER_VALIDATOR_SOURCE_MISSING:{path}")
        files.append(
            {"path": path, "sha256": "sha256:" + hashlib.sha256(result.stdout).hexdigest()}
        )
    return {"source_files": files, "source_sha256": sha256(files)}


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _task_counts(row: dict[str, Any]) -> tuple[int, int, int]:
    operator = len(row["oracle_work_plan"])
    arg1 = sum(step[0] != "END" for step in row["oracle_work_plan"])
    arg2 = sum(step[0] in {"UNSTACK", "STACK"} for step in row["oracle_work_plan"])
    return operator, arg1, arg2


def _float32_total(operator: float, arg1: float | None, arg2: float | None) -> float:
    total = torch.tensor(operator, dtype=torch.float32)
    if arg1 is not None:
        total = total + torch.tensor(arg1, dtype=torch.float32)
    if arg2 is not None:
        total = total + torch.tensor(arg2, dtype=torch.float32)
    return float(total)


def _validate_update_schedule(
    result: dict[str, Any], row_by_id: dict[str, dict[str, Any]]
) -> None:
    arm = result["arm"]
    seed = int(result["seed"])
    order = list(ARMS[arm])
    if result.get("task_order") != order:
        raise ValueError(f"A2_ORDER_VALIDATOR_ARM_ORDER_METADATA:{arm}:{seed}")
    updates = result.get("updates")
    if not isinstance(updates, list) or len(updates) != MAX_EPOCH * len(TASKS):
        raise ValueError(f"A2_ORDER_VALIDATOR_UPDATE_COUNT:{arm}:{seed}")
    for index, update in enumerate(updates):
        expected_task = order[index % 3]
        expected_epoch = index // 3
        if (
            update.get("update_index") != index
            or update.get("epoch_index") != expected_epoch
            or update.get("task_id") != expected_task
        ):
            raise ValueError(f"A2_ORDER_VALIDATOR_UPDATE_SCHEDULE:{arm}:{seed}:{index}")
        operator_count, arg1_count, arg2_count = _task_counts(row_by_id[expected_task])
        if (
            update.get("operator_target_count") != operator_count
            or update.get("arg1_target_count") != arg1_count
            or update.get("arg2_target_count") != arg2_count
            or update.get("operator_position_weight") != 1.0 / operator_count
        ):
            raise ValueError(f"A2_ORDER_VALIDATOR_UPDATE_TARGETS:{arm}:{seed}:{index}")

        operator_loss = float(update["operator_loss"])
        arg1_loss = update.get("arg1_pointer_loss")
        arg2_loss = update.get("arg2_pointer_loss")
        if arg1_count == 0:
            if arg1_loss is not None:
                raise ValueError(f"A2_ORDER_VALIDATOR_ARG1_APPLICABILITY:{arm}:{seed}:{index}")
        elif arg1_loss is None or not math.isfinite(float(arg1_loss)):
            raise ValueError(f"A2_ORDER_VALIDATOR_ARG1_APPLICABILITY:{arm}:{seed}:{index}")
        if arg2_count == 0:
            if arg2_loss is not None:
                raise ValueError(f"A2_ORDER_VALIDATOR_ARG2_APPLICABILITY:{arm}:{seed}:{index}")
        elif arg2_loss is None or not math.isfinite(float(arg2_loss)):
            raise ValueError(f"A2_ORDER_VALIDATOR_ARG2_APPLICABILITY:{arm}:{seed}:{index}")
        if not math.isfinite(operator_loss) or not math.isfinite(float(update["total_loss"])):
            raise ValueError(f"A2_ORDER_VALIDATOR_NONFINITE_LOSS:{arm}:{seed}:{index}")
        expected_total = _float32_total(
            operator_loss,
            float(arg1_loss) if arg1_loss is not None else None,
            float(arg2_loss) if arg2_loss is not None else None,
        )
        if float(update["total_loss"]) != expected_total:
            raise ValueError(f"A2_ORDER_VALIDATOR_LOSS_DECOMPOSITION:{arm}:{seed}:{index}")

        grad = float(update["gradient_norm"])
        clip = float(update["gradient_clip_norm"])
        if not math.isfinite(grad) or clip != 1.0:
            raise ValueError(f"A2_ORDER_VALIDATOR_GRADIENT:{arm}:{seed}:{index}")
        if update.get("clipping_occurred") != (grad > clip):
            raise ValueError(f"A2_ORDER_VALIDATOR_CLIPPING:{arm}:{seed}:{index}")


def _validate_position0_record(
    item: dict[str, Any], row: dict[str, Any], *, arm: str, seed: int, epoch: int
) -> None:
    gold = row["oracle_work_plan"][0][0]
    task_id = row["task_id"]
    if item.get("task_id") != task_id or item.get("gold_operator") != gold:
        raise ValueError(f"A2_ORDER_VALIDATOR_P0_GOLD:{arm}:{seed}:{epoch}:{task_id}")
    if item.get("operator_correct") != (item.get("predicted_operator") == gold):
        raise ValueError(f"A2_ORDER_VALIDATOR_P0_CORRECT:{arm}:{seed}:{epoch}:{task_id}")
    p_gold = float(item["probability_gold_operator"])
    p_end = float(item["probability_end"])
    if not 0.0 <= p_gold <= 1.0 or not 0.0 <= p_end <= 1.0:
        raise ValueError(f"A2_ORDER_VALIDATOR_P0_PROB:{arm}:{seed}:{epoch}:{task_id}")
    expected_nll = -math.log(max(p_gold, torch.finfo(torch.float32).tiny))
    if float(item["operator_nll"]) != expected_nll:
        raise ValueError(f"A2_ORDER_VALIDATOR_P0_NLL:{arm}:{seed}:{epoch}:{task_id}")


def _free_goal_success(row: dict[str, Any], predicted_plan: Any) -> tuple[bool, bool, int]:
    task = task_from_row(row)
    state = validate_state(task.blocks, task.initial)
    initial = goal_satisfied(state, task.goal)
    if not isinstance(predicted_plan, list) or not predicted_plan or predicted_plan[-1] != ["END"]:
        return initial, False, 0
    action_steps = predicted_plan[:-1]
    for step in action_steps:
        try:
            action = parse_nonterminal_step(step, list(task.blocks))
            state = apply_action(task.blocks, state, action)
        except (ValueError, TypeError, IndexError):
            return initial, False, len(action_steps)
    executable = bool(action_steps) or initial
    return initial, bool(executable and goal_satisfied(state, task.goal)), len(action_steps)


def _validate_free_record(
    item: dict[str, Any], row: dict[str, Any], *, arm: str, seed: int, epoch: int
) -> None:
    task_id = row["task_id"]
    if item.get("task_id") != task_id:
        raise ValueError(f"A2_ORDER_VALIDATOR_FREE_TASK:{arm}:{seed}:{epoch}:{task_id}")
    initial, success, length = _free_goal_success(row, item.get("predicted_plan"))
    if item.get("initial_goal_satisfied") != initial:
        raise ValueError(f"A2_ORDER_VALIDATOR_FREE_INITIAL:{arm}:{seed}:{epoch}:{task_id}")
    if item.get("final_goal_success") != success:
        raise ValueError(f"A2_ORDER_VALIDATOR_FREE_GOAL:{arm}:{seed}:{epoch}:{task_id}")
    if item.get("predicted_plan_length") != length:
        raise ValueError(f"A2_ORDER_VALIDATOR_FREE_LENGTH:{arm}:{seed}:{epoch}:{task_id}")
    if item.get("exact_plan_match") != (item.get("predicted_plan") == row["oracle_work_plan"]):
        raise ValueError(f"A2_ORDER_VALIDATOR_FREE_EXACT:{arm}:{seed}:{epoch}:{task_id}")


def _validate_epoch_evidence(
    result: dict[str, Any], row_by_id: dict[str, dict[str, Any]]
) -> None:
    arm = result["arm"]
    seed = int(result["seed"])
    records = result.get("epoch_evidence")
    if not isinstance(records, list) or len(records) != MAX_EPOCH:
        raise ValueError(f"A2_ORDER_VALIDATOR_EPOCH_COVERAGE:{arm}:{seed}")
    for expected_epoch, record in enumerate(records, 1):
        epoch = int(record.get("epoch"))
        if epoch != expected_epoch or record.get("update_count") != epoch * 3:
            raise ValueError(f"A2_ORDER_VALIDATOR_EPOCH_INDEX:{arm}:{seed}:{expected_epoch}")
        p0 = record.get("position0")
        free = record.get("free_running")
        if not isinstance(p0, list) or not isinstance(free, list) or len(p0) != 3 or len(free) != 3:
            raise ValueError(f"A2_ORDER_VALIDATOR_EPOCH_SHAPE:{arm}:{seed}:{epoch}")
        p0_by_id = {item.get("task_id"): item for item in p0}
        free_by_id = {item.get("task_id"): item for item in free}
        if set(p0_by_id) != set(TASKS) or len(p0_by_id) != 3:
            raise ValueError(f"A2_ORDER_VALIDATOR_EPOCH_P0_COVERAGE:{arm}:{seed}:{epoch}")
        if set(free_by_id) != set(TASKS) or len(free_by_id) != 3:
            raise ValueError(f"A2_ORDER_VALIDATOR_EPOCH_FREE_COVERAGE:{arm}:{seed}:{epoch}")
        for task_id in TASKS:
            _validate_position0_record(
                p0_by_id[task_id], row_by_id[task_id], arm=arm, seed=seed, epoch=epoch
            )
            _validate_free_record(
                free_by_id[task_id], row_by_id[task_id], arm=arm, seed=seed, epoch=epoch
            )


def _position0_rescued(record: dict[str, Any]) -> bool:
    by_id = {item["task_id"]: item for item in record["position0"]}
    return all(
        by_id[task_id]["gold_operator"] == "UNSTACK"
        and bool(by_id[task_id]["operator_correct"])
        for task_id in NONTRIVIAL
    )


def _free_rescued(record: dict[str, Any]) -> bool:
    by_id = {item["task_id"]: item for item in record["free_running"]}
    return all(
        by_id[task_id]["initial_goal_satisfied"] is False
        and bool(by_id[task_id]["final_goal_success"])
        for task_id in NONTRIVIAL
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
    position0_by_task = {
        task["task_id"]: {
            "gold_operator": task["positions"][0]["gold_operator"],
            "predicted_operator": task["positions"][0]["predicted_operator"],
            "operator_correct": task["positions"][0]["operator_correct"],
            "probability_gold_operator": task["positions"][0]["probability_gold_operator"],
            "operator_nll": task["positions"][0]["operator_nll"],
            "probability_end": task["positions"][0]["probability_end"],
        }
        for task in tasks
    }
    end_probs = {
        task_id: float(item["probability_end"])
        for task_id, item in position0_by_task.items()
    }
    nontrivial_mean = (end_probs["bw-00000002"] + end_probs["bw-00000003"]) / 2
    per_task = {}
    for task in tasks:
        p = task["positions"]
        p_non = [row for row in p if row["gold_operator"] != "END"]
        p_end = [row for row in p if row["gold_operator"] == "END"]
        p_arg1 = [row for row in p if row["has_arg1_target"]]
        p_arg2 = [row for row in p if row["has_arg2_target"]]
        per_task[task["task_id"]] = {
            "operator_accuracy": _rate([row["operator_correct"] for row in p]),
            "non_end_operator_accuracy": _rate([row["operator_correct"] for row in p_non]),
            "end_accuracy": _rate([row["operator_correct"] for row in p_end]),
            "arg1_accuracy": _rate([bool(row["arg1_correct"]) for row in p_arg1]),
            "arg2_accuracy": _rate([bool(row["arg2_correct"]) for row in p_arg2]),
            "joint_step_accuracy": _rate([row["joint_step_correct"] for row in p]),
        }
    return {
        "aggregate": {
            "operator_accuracy": _rate([row["operator_correct"] for row in positions]),
            "non_end_operator_accuracy": _rate([row["operator_correct"] for row in non_end]),
            "end_accuracy": _rate([row["operator_correct"] for row in end]),
            "arg1_accuracy": _rate([bool(row["arg1_correct"]) for row in arg1]),
            "arg2_accuracy": _rate([bool(row["arg2_correct"]) for row in arg2]),
            "joint_step_accuracy": _rate([row["joint_step_correct"] for row in positions]),
            "predicted_end_rate": _rate([row["predicted_operator"] == "END" for row in positions]),
        },
        "per_task": per_task,
        "position0_by_task": position0_by_task,
        "position0_unstack": {
            "target_count": len(pos0_unstack),
            "accuracy": _rate([row["operator_correct"] for row in pos0_unstack]),
            "mean_gold_operator_probability": _mean(
                [float(row["probability_gold_operator"]) for row in pos0_unstack]
            ),
            "mean_operator_nll": _mean([float(row["operator_nll"]) for row in pos0_unstack]),
            "mean_end_probability": _mean([float(row["probability_end"]) for row in pos0_unstack]),
        },
        "position4_end": {
            "target_count": len(pos4_end),
            "accuracy": _rate([row["operator_correct"] for row in pos4_end]),
            "mean_gold_operator_probability": _mean(
                [float(row["probability_gold_operator"]) for row in pos4_end]
            ),
            "mean_operator_nll": _mean([float(row["operator_nll"]) for row in pos4_end]),
            "mean_end_probability": _mean([float(row["probability_end"]) for row in pos4_end]),
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


def _validate_teacher_tasks(
    tasks: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    arm: str,
    seed: int,
    epoch: int,
) -> None:
    by_id = {task.get("task_id"): task for task in tasks}
    if len(tasks) != 3 or set(by_id) != set(TASKS):
        raise ValueError(f"A2_ORDER_VALIDATOR_CHECKPOINT_TEACHER_COVERAGE:{arm}:{seed}:{epoch}")
    for task_id in TASKS:
        task = by_id[task_id]
        if task.get("seed") != seed or task.get("split") != "train":
            raise ValueError(f"A2_ORDER_VALIDATOR_CHECKPOINT_TEACHER_SCOPE:{arm}:{seed}:{epoch}:{task_id}")
        gold_plan = row_by_id[task_id]["oracle_work_plan"]
        positions = task.get("positions")
        if not isinstance(positions, list) or len(positions) != len(gold_plan):
            raise ValueError(f"A2_ORDER_VALIDATOR_CHECKPOINT_TEACHER_POSITIONS:{arm}:{seed}:{epoch}:{task_id}")
        for index, (position, gold_step) in enumerate(zip(positions, gold_plan, strict=True)):
            gold_operator = gold_step[0]
            gold_arg1 = gold_step[1] if len(gold_step) > 1 else None
            gold_arg2 = gold_step[2] if len(gold_step) > 2 else None
            if (
                position.get("position_index") != index
                or position.get("gold_operator") != gold_operator
                or position.get("gold_arg1") != gold_arg1
                or position.get("gold_arg2") != gold_arg2
            ):
                raise ValueError(f"A2_ORDER_VALIDATOR_CHECKPOINT_TEACHER_GOLD:{arm}:{seed}:{epoch}:{task_id}:{index}")
            operator_correct = position.get("predicted_operator") == gold_operator
            arg1_correct = (
                position.get("arg1_head_prediction") == gold_arg1 if gold_arg1 is not None else None
            )
            arg2_correct = (
                position.get("arg2_head_prediction") == gold_arg2 if gold_arg2 is not None else None
            )
            joint = operator_correct and arg1_correct is not False and arg2_correct is not False
            if (
                position.get("operator_correct") != operator_correct
                or position.get("arg1_correct") != arg1_correct
                or position.get("arg2_correct") != arg2_correct
                or position.get("joint_step_correct") != joint
            ):
                raise ValueError(f"A2_ORDER_VALIDATOR_CHECKPOINT_TEACHER_CLAIM:{arm}:{seed}:{epoch}:{task_id}:{index}")


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


def _validate_checkpoints(
    result: dict[str, Any], row_by_id: dict[str, dict[str, Any]]
) -> None:
    arm = result["arm"]
    seed = int(result["seed"])
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list) or [item.get("epoch") for item in checkpoints] != list(CHECKPOINTS):
        raise ValueError(f"A2_ORDER_VALIDATOR_CHECKPOINT_COVERAGE:{arm}:{seed}")
    for checkpoint in checkpoints:
        epoch = int(checkpoint["epoch"])
        if checkpoint.get("update_count") != epoch * 3:
            raise ValueError(f"A2_ORDER_VALIDATOR_CHECKPOINT_UPDATE_COUNT:{arm}:{seed}:{epoch}")
        teacher = checkpoint.get("teacher_forced")
        free = checkpoint.get("free_running")
        _validate_teacher_tasks(teacher, row_by_id, arm=arm, seed=seed, epoch=epoch)
        if checkpoint.get("teacher_forced_summary") != _teacher_summary(teacher):
            raise ValueError(f"A2_ORDER_VALIDATOR_CHECKPOINT_TEACHER_SUMMARY:{arm}:{seed}:{epoch}")
        free_by_id = {task.get("task_id"): task for task in free}
        if len(free) != 3 or set(free_by_id) != set(TASKS):
            raise ValueError(f"A2_ORDER_VALIDATOR_CHECKPOINT_FREE_COVERAGE:{arm}:{seed}:{epoch}")
        for task_id in TASKS:
            task = free_by_id[task_id]
            _validate_free_record(
                {
                    "task_id": task_id,
                    "initial_goal_satisfied": task["initial_goal_satisfied"],
                    "predicted_plan": task["predicted_plan"],
                    "predicted_plan_length": task["predicted_plan_length"],
                    "exact_plan_match": task["exact_plan_match"],
                    "final_goal_success": task["final_goal_success"],
                },
                row_by_id[task_id],
                arm=arm,
                seed=seed,
                epoch=epoch,
            )
        if checkpoint.get("free_running_summary") != _free_summary(free):
            raise ValueError(f"A2_ORDER_VALIDATOR_CHECKPOINT_FREE_SUMMARY:{arm}:{seed}:{epoch}")


def _validate_rescue_claims(result: dict[str, Any]) -> None:
    arm = result["arm"]
    seed = int(result["seed"])
    records = result["epoch_evidence"]
    p0 = _first(records, _position0_rescued)
    free = _first(records, _free_rescued)
    expected_events = {
        "first_position0_operator_rescue": p0,
        "first_full_free_running_rescue": free,
    }
    if result.get("rescue_events") != expected_events:
        raise ValueError(f"A2_ORDER_VALIDATOR_RESCUE_EVENT:{arm}:{seed}")
    expected_persistence = {
        "position0_operator_rescue": _persistence(records, p0, _position0_rescued),
        "full_free_running_rescue": _persistence(records, free, _free_rescued),
    }
    if result.get("rescue_persistence") != expected_persistence:
        raise ValueError(f"A2_ORDER_VALIDATOR_RESCUE_PERSISTENCE:{arm}:{seed}")


def _actual_prefix(result: dict[str, Any]) -> dict[str, Any]:
    checkpoint = next(item for item in result["checkpoints"] if item["epoch"] == 3)
    return {
        "initialization_canonical_sha256": result["initialization_canonical_sha256"],
        "trained_canonical_sha256": checkpoint["trained_canonical_sha256"],
        "optimizer_canonical_sha256": checkpoint["optimizer_canonical_sha256"],
        "updates": [
            {field: update[field] for field in PREFIX_TRACE_FIELDS}
            for update in result["updates"][:9]
        ],
    }


def _frozen_control_projection(
    rows: list[dict[str, Any]], *, seed: int, dataset_hash: str
) -> dict[str, Any]:
    """Reconstruct the real frozen historical 3-epoch A2 control independently."""
    configure_canonical_cpu_runtime(seed)
    with tempfile.TemporaryDirectory(prefix="a2-order-validator-control-") as temp:
        _model, checkpoint, trace = _train_a2_with_loss_trace(
            rows, seed, Path(temp), dataset_hash
        )
    updates = []
    for update in trace:
        operator_count = int(update["operator_target_count"])
        enriched = {**update, "operator_position_weight": 1.0 / operator_count}
        updates.append({field: enriched[field] for field in PREFIX_TRACE_FIELDS})
    return {
        "initialization_canonical_sha256": checkpoint[
            "canonical_initialization_state_dict_sha256"
        ],
        "trained_canonical_sha256": checkpoint["canonical_trained_state_dict_sha256"],
        "optimizer_canonical_sha256": checkpoint["canonical_optimizer_state_sha256"],
        "updates": updates,
    }


def _validate_prefix(
    result: dict[str, Any], frozen_control: dict[str, Any] | None
) -> None:
    arm = result["arm"]
    seed = int(result["seed"])
    record = result.get("prefix_equivalence")
    if arm != "canonical_order":
        if record is not None:
            raise ValueError(f"A2_ORDER_VALIDATOR_NONCANONICAL_PREFIX:{arm}:{seed}")
        return
    if frozen_control is None:
        raise ValueError(f"A2_ORDER_VALIDATOR_FROZEN_CONTROL_MISSING:{seed}")
    if not isinstance(record, dict) or record.get("status") != "PASS":
        raise ValueError(f"A2_ORDER_VALIDATOR_CANONICAL_PREFIX_STATUS:{seed}")
    if record.get("seed") != seed or record.get("trace_fields") != list(PREFIX_TRACE_FIELDS):
        raise ValueError(f"A2_ORDER_VALIDATOR_CANONICAL_PREFIX_METADATA:{seed}")
    actual = _actual_prefix(result)
    if actual != frozen_control:
        raise ValueError(f"A2_ORDER_VALIDATOR_FROZEN_CONTROL_ANCHOR:{seed}")
    if record.get("arm_prefix") != actual or record.get("control") != frozen_control:
        raise ValueError(f"A2_ORDER_VALIDATOR_CANONICAL_PREFIX_BINDING:{seed}")


def _event_update(result: dict[str, Any], key: str) -> int | None:
    event = result["rescue_events"][key]
    return int(event["update_count"]) if event is not None else None


def _recompute_arm_summaries(results: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for arm in ARMS:
        arm_results = sorted(
            [result for result in results if result["arm"] == arm],
            key=lambda result: int(result["seed"]),
        )
        p0 = [_event_update(result, "first_position0_operator_rescue") for result in arm_results]
        free = [_event_update(result, "first_full_free_running_rescue") for result in arm_results]

        def mean_complete(values: list[int | None]) -> float | None:
            return None if any(value is None for value in values) else sum(
                int(value) for value in values if value is not None
            ) / len(values)

        output[arm] = {
            "seed_count": len(arm_results),
            "task_order": list(ARMS[arm]),
            "first_position0_operator_rescue_update_by_seed": {
                str(result["seed"]): _event_update(result, "first_position0_operator_rescue")
                for result in arm_results
            },
            "first_position0_operator_rescue_mean_update_if_all_seeds": mean_complete(p0),
            "first_full_free_running_rescue_update_by_seed": {
                str(result["seed"]): _event_update(result, "first_full_free_running_rescue")
                for result in arm_results
            },
            "first_full_free_running_rescue_mean_update_if_all_seeds": mean_complete(free),
        }
    return output


def _recompute_deltas(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(result["arm"], int(result["seed"])): result for result in results}
    output = {}
    for arm in ("task01_middle", "task01_last"):
        by_seed = {}
        for seed in SEEDS:
            canonical = by_key[("canonical_order", seed)]
            other = by_key[(arm, seed)]
            record = {}
            for key in (
                "first_position0_operator_rescue",
                "first_full_free_running_rescue",
            ):
                left = _event_update(canonical, key)
                right = _event_update(other, key)
                record[f"{key}_update_delta_vs_canonical"] = (
                    right - left if left is not None and right is not None else None
                )
            by_seed[str(seed)] = record
        output[arm] = {"by_seed": by_seed}
    return output


def validate_claims_from_evidence(
    payload: dict[str, Any], *, implementation_commit: str
) -> dict[str, Any]:
    if payload.get("implementation_commit") != implementation_commit:
        raise ValueError("A2_ORDER_VALIDATOR_IMPLEMENTATION_MISMATCH")
    expected_source = _source_identity_at_commit(implementation_commit)
    if payload.get("source_files") != expected_source["source_files"]:
        raise ValueError("A2_ORDER_VALIDATOR_SOURCE_FILES")
    if payload.get("source_sha256") != expected_source["source_sha256"]:
        raise ValueError("A2_ORDER_VALIDATOR_SOURCE_IDENTITY")
    if payload.get("heldout_accessed") is not False:
        raise ValueError("A2_ORDER_VALIDATOR_HELDOUT")
    if payload.get("go_latent") != "NOT EVALUATED":
        raise ValueError("A2_ORDER_VALIDATOR_GO_LATENT")
    if payload.get("seeds") != list(SEEDS):
        raise ValueError("A2_ORDER_VALIDATOR_SEEDS")
    if payload.get("arms") != {name: list(order) for name, order in ARMS.items()}:
        raise ValueError("A2_ORDER_VALIDATOR_ARMS")
    if payload.get("checkpoint_epochs") != list(CHECKPOINTS) or payload.get("max_epoch") != MAX_EPOCH:
        raise ValueError("A2_ORDER_VALIDATOR_BUDGET")
    unsigned = {key: value for key, value in payload.items() if key != "canonical_identity"}
    if payload.get("canonical_identity") != sha256(unsigned):
        raise ValueError("A2_ORDER_VALIDATOR_CANONICAL_IDENTITY")

    dataset = generate_train_only()
    rows = sorted(list(dataset["train"]), key=lambda row: row["task_id"])
    row_by_id = {row["task_id"]: row for row in rows}
    if set(row_by_id) != set(TASKS) or len(row_by_id) != 3:
        raise ValueError("A2_ORDER_VALIDATOR_DATASET_SCOPE")
    if payload.get("dataset", {}).get("evaluated_task_ids") != list(TASKS):
        raise ValueError("A2_ORDER_VALIDATOR_EVALUATED_TASKS")
    dataset_hash = dataset["frozen_dataset_lineage_hash"]
    if payload.get("dataset", {}).get("frozen_dataset_lineage_hash") != dataset_hash:
        raise ValueError("A2_ORDER_VALIDATOR_DATASET_HASH")

    results = payload.get("arm_seed_results")
    if not isinstance(results, list) or len(results) != len(ARMS) * len(SEEDS):
        raise ValueError("A2_ORDER_VALIDATOR_RESULT_COUNT")
    keys = [(result.get("arm"), result.get("seed")) for result in results]
    expected_keys = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(keys) != expected_keys or len(keys) != len(set(keys)):
        raise ValueError("A2_ORDER_VALIDATOR_ARM_SEED_COVERAGE")

    frozen_by_seed = {
        seed: _frozen_control_projection(rows, seed=seed, dataset_hash=dataset_hash)
        for seed in SEEDS
    }
    init_by_seed: dict[int, str] = {}
    for result in results:
        seed = int(result["seed"])
        init = result["initialization_canonical_sha256"]
        if seed in init_by_seed and init_by_seed[seed] != init:
            raise ValueError(f"A2_ORDER_VALIDATOR_INITIALIZATION_POLICY:{seed}")
        init_by_seed[seed] = init
        _validate_update_schedule(result, row_by_id)
        _validate_epoch_evidence(result, row_by_id)
        _validate_checkpoints(result, row_by_id)
        _validate_rescue_claims(result)
        _validate_prefix(
            result,
            frozen_by_seed[seed] if result["arm"] == "canonical_order" else None,
        )

    summaries = _recompute_arm_summaries(results)
    if payload.get("cross_seed_arm_summaries") != summaries:
        raise ValueError("A2_ORDER_VALIDATOR_CROSS_SEED_SUMMARY")
    deltas = _recompute_deltas(results)
    if payload.get("cross_arm_rescue_deltas") != deltas:
        raise ValueError("A2_ORDER_VALIDATOR_CROSS_ARM_DELTA")

    return {
        "independent_claim_validation": "PASS",
        "arm_seed_coverage": len(results),
        "source_binding": "PASS",
        "canonical_prefix_equivalence": "PASS",
        "frozen_control_anchor": "PASS",
        "loss_decomposition_validation": "PASS",
        "rescue_reconstruction": "PASS",
        "cross_arm_delta_reconstruction": "PASS",
    }
