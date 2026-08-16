"""Independent persisted-evidence validator for the A2 optimization-budget trajectory."""

from __future__ import annotations

import math
from typing import Any

from .dataset import task_from_row
from .domain import apply_action, goal_satisfied, validate_state
from .e2e import parse_nonterminal_step
from .train_only_dataset import FROZEN_DATASET_LINEAGE_HASH_V1, generate_train_only

SEEDS = (17, 29, 43)
EXPECTED_TRAIN_TASK_IDS = ("bw-00000001", "bw-00000002", "bw-00000003")
CHECKPOINT_EPOCHS = (3, 10, 30, 100)
MAX_EPOCH = 100
HELDOUT = ("bw-00000004", "bw-00000005")
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


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _task_counts(row: dict[str, Any]) -> tuple[int, int, int]:
    operator_count = len(row["oracle_work_plan"])
    arg1_count = sum(step[0] != "END" for step in row["oracle_work_plan"])
    arg2_count = sum(step[0] in {"UNSTACK", "STACK"} for step in row["oracle_work_plan"])
    return operator_count, arg1_count, arg2_count


def _validate_training_schedule(
    training: dict[str, Any], row_by_id: dict[str, dict[str, Any]], *, seed: int
) -> None:
    updates = training["updates"]
    if len(updates) != MAX_EPOCH * len(EXPECTED_TRAIN_TASK_IDS):
        raise ValueError(f"A2_BUDGET_VALIDATOR_UPDATE_COUNT:{seed}")
    expected_order = list(EXPECTED_TRAIN_TASK_IDS)
    for index, update in enumerate(updates):
        epoch_index = index // len(expected_order)
        task_id = expected_order[index % len(expected_order)]
        if update.get("update_index") != index:
            raise ValueError(f"A2_BUDGET_VALIDATOR_UPDATE_INDEX:{seed}:{index}")
        if update.get("epoch_index") != epoch_index or update.get("task_id") != task_id:
            raise ValueError(f"A2_BUDGET_VALIDATOR_UPDATE_SCHEDULE:{seed}:{index}")
        operator_count, arg1_count, arg2_count = _task_counts(row_by_id[task_id])
        if update.get("operator_target_count") != operator_count:
            raise ValueError(f"A2_BUDGET_VALIDATOR_OPERATOR_COUNT:{seed}:{index}")
        if update.get("arg1_target_count") != arg1_count:
            raise ValueError(f"A2_BUDGET_VALIDATOR_ARG1_COUNT:{seed}:{index}")
        if update.get("arg2_target_count") != arg2_count:
            raise ValueError(f"A2_BUDGET_VALIDATOR_ARG2_COUNT:{seed}:{index}")
        if update.get("operator_position_weight") != 1.0 / operator_count:
            raise ValueError(f"A2_BUDGET_VALIDATOR_OPERATOR_WEIGHT:{seed}:{index}")
        gradient_norm = float(update["gradient_norm"])
        clip_norm = float(update["gradient_clip_norm"])
        if clip_norm != 1.0 or update.get("clipping_occurred") != (gradient_norm > clip_norm):
            raise ValueError(f"A2_BUDGET_VALIDATOR_CLIPPING:{seed}:{index}")
        for field in ("operator_loss", "total_loss", "gradient_norm"):
            if not math.isfinite(float(update[field])):
                raise ValueError(f"A2_BUDGET_VALIDATOR_NONFINITE:{seed}:{index}:{field}")


def _project_actual_prefix(training: dict[str, Any]) -> dict[str, Any]:
    checkpoint = next(
        (item for item in training["checkpoints"] if item.get("epoch") == 3),
        None,
    )
    if checkpoint is None:
        raise ValueError("A2_BUDGET_VALIDATOR_PREFIX_CHECKPOINT_MISSING")
    return {
        "initialization_canonical_sha256": training["initialization_canonical_sha256"],
        "trained_canonical_sha256": checkpoint["trained_canonical_sha256"],
        "optimizer_canonical_sha256": checkpoint["optimizer_canonical_sha256"],
        "updates": [
            {field: update[field] for field in PREFIX_TRACE_FIELDS}
            for update in training["updates"][:9]
        ],
    }


def _validate_prefix_equivalence(
    record: dict[str, Any], training: dict[str, Any], *, seed: int
) -> None:
    if record.get("seed") != seed or record.get("status") != "PASS":
        raise ValueError(f"A2_BUDGET_VALIDATOR_PREFIX_STATUS:{seed}")
    if record.get("purpose") != "NON_SCIENTIFIC_FROZEN_3_EPOCH_PREFIX_EQUIVALENCE":
        raise ValueError(f"A2_BUDGET_VALIDATOR_PREFIX_PURPOSE:{seed}")
    if record.get("trace_fields") != list(PREFIX_TRACE_FIELDS):
        raise ValueError(f"A2_BUDGET_VALIDATOR_PREFIX_FIELDS:{seed}")
    control = record.get("control")
    prefix = record.get("trajectory_prefix")
    actual_prefix = _project_actual_prefix(training)
    if prefix != actual_prefix:
        raise ValueError(f"A2_BUDGET_VALIDATOR_PREFIX_TRAJECTORY_BINDING:{seed}")
    if control != actual_prefix:
        raise ValueError(f"A2_BUDGET_VALIDATOR_PREFIX_CONTROL_BINDING:{seed}")
    if len(control["updates"]) != 9:
        raise ValueError(f"A2_BUDGET_VALIDATOR_PREFIX_UPDATE_COUNT:{seed}")
    expected = list(EXPECTED_TRAIN_TASK_IDS) * 3
    if [item["task_id"] for item in control["updates"]] != expected:
        raise ValueError(f"A2_BUDGET_VALIDATOR_PREFIX_SCHEDULE:{seed}")


def _validate_teacher_raw(
    tasks: list[dict[str, Any]], row_by_id: dict[str, dict[str, Any]], *, seed: int
) -> None:
    if len(tasks) != len(row_by_id):
        raise ValueError(f"A2_BUDGET_VALIDATOR_TEACHER_TASK_COUNT:{seed}")
    seen = set()
    for task in tasks:
        task_id = task.get("task_id")
        if task_id in seen or task_id not in row_by_id:
            raise ValueError(f"A2_BUDGET_VALIDATOR_TEACHER_TASK_ID:{seed}")
        seen.add(task_id)
        if task.get("seed") != seed or task.get("split") != "train":
            raise ValueError(f"A2_BUDGET_VALIDATOR_TEACHER_SCOPE:{seed}:{task_id}")
        row = row_by_id[task_id]
        positions = task.get("positions")
        if len(positions) != len(row["oracle_work_plan"]):
            raise ValueError(f"A2_BUDGET_VALIDATOR_TEACHER_POSITION_COUNT:{seed}:{task_id}")
        for index, (position, step) in enumerate(
            zip(positions, row["oracle_work_plan"], strict=True)
        ):
            gold_operator = step[0]
            gold_arg1 = step[1] if len(step) > 1 else None
            gold_arg2 = step[2] if len(step) > 2 else None
            if position.get("position_index") != index:
                raise ValueError(
                    f"A2_BUDGET_VALIDATOR_TEACHER_POSITION_INDEX:{seed}:{task_id}:{index}"
                )
            if (
                position.get("gold_operator") != gold_operator
                or position.get("gold_arg1") != gold_arg1
                or position.get("gold_arg2") != gold_arg2
            ):
                raise ValueError(
                    f"A2_BUDGET_VALIDATOR_TEACHER_GOLD:{seed}:{task_id}:{index}"
                )
            expected_arg1 = gold_arg1 is not None
            expected_arg2 = gold_arg2 is not None
            if (
                position.get("has_arg1_target") != expected_arg1
                or position.get("has_arg2_target") != expected_arg2
            ):
                raise ValueError(
                    f"A2_BUDGET_VALIDATOR_TEACHER_TARGET_FLAGS:{seed}:{task_id}:{index}"
                )
            operator_correct = position.get("predicted_operator") == gold_operator
            arg1_correct = (
                position.get("arg1_head_prediction") == gold_arg1 if expected_arg1 else None
            )
            arg2_correct = (
                position.get("arg2_head_prediction") == gold_arg2 if expected_arg2 else None
            )
            joint = operator_correct and arg1_correct is not False and arg2_correct is not False
            if position.get("operator_correct") != operator_correct:
                raise ValueError(
                    f"A2_BUDGET_VALIDATOR_TEACHER_OPERATOR_CLAIM:{seed}:{task_id}:{index}"
                )
            if position.get("arg1_correct") != arg1_correct:
                raise ValueError(
                    f"A2_BUDGET_VALIDATOR_TEACHER_ARG1_CLAIM:{seed}:{task_id}:{index}"
                )
            if position.get("arg2_correct") != arg2_correct:
                raise ValueError(
                    f"A2_BUDGET_VALIDATOR_TEACHER_ARG2_CLAIM:{seed}:{task_id}:{index}"
                )
            if position.get("joint_step_correct") != joint:
                raise ValueError(
                    f"A2_BUDGET_VALIDATOR_TEACHER_JOINT_CLAIM:{seed}:{task_id}:{index}"
                )
            probability_gold = float(position["probability_gold_operator"])
            probability_end = float(position["probability_end"])
            if not 0.0 <= probability_gold <= 1.0 or not 0.0 <= probability_end <= 1.0:
                raise ValueError(
                    f"A2_BUDGET_VALIDATOR_TEACHER_PROBABILITY:{seed}:{task_id}:{index}"
                )
            expected_nll = -math.log(max(probability_gold, 1.1754943508222875e-38))
            if float(position["operator_nll"]) != expected_nll:
                raise ValueError(
                    f"A2_BUDGET_VALIDATOR_TEACHER_NLL:{seed}:{task_id}:{index}"
                )
    if seen != set(row_by_id):
        raise ValueError(f"A2_BUDGET_VALIDATOR_TEACHER_COVERAGE:{seed}")


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


def _teacher_summary_from_raw(tasks: list[dict[str, Any]]) -> dict[str, Any]:
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


def _recompute_free_claims(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    task = task_from_row(row)
    raw = item["predicted_plan"]
    state = validate_state(task.blocks, task.initial)
    initial_goal_satisfied = goal_satisfied(state, task.goal)
    terminal_end = bool(raw and raw[-1] == ["END"])
    action_steps = raw[:-1] if terminal_end else raw
    executable = True
    for step in action_steps:
        try:
            action = parse_nonterminal_step(step, list(task.blocks))
            state = apply_action(task.blocks, state, action)
        except (TypeError, ValueError):
            executable = False
            break
    predicted_plan_length = len(action_steps)
    full_plan_executable = terminal_end and executable and (
        predicted_plan_length > 0 or initial_goal_satisfied
    )
    return {
        "initial_goal_satisfied": initial_goal_satisfied,
        "predicted_plan_length": predicted_plan_length,
        "exact_plan_match": raw == row["oracle_work_plan"],
        "final_goal_success": full_plan_executable and goal_satisfied(state, task.goal),
    }


def _validate_free_raw(
    tasks: list[dict[str, Any]], row_by_id: dict[str, dict[str, Any]], *, seed: int
) -> None:
    if len(tasks) != len(row_by_id):
        raise ValueError(f"A2_BUDGET_VALIDATOR_FREE_TASK_COUNT:{seed}")
    seen = set()
    for item in tasks:
        task_id = item.get("task_id")
        if task_id in seen or task_id not in row_by_id:
            raise ValueError(f"A2_BUDGET_VALIDATOR_FREE_TASK_ID:{seed}")
        seen.add(task_id)
        if item.get("seed") != seed or item.get("split") != "train":
            raise ValueError(f"A2_BUDGET_VALIDATOR_FREE_SCOPE:{seed}:{task_id}")
        recomputed = _recompute_free_claims(row_by_id[task_id], item)
        for field, expected in recomputed.items():
            if item.get(field) != expected:
                raise ValueError(
                    f"A2_BUDGET_VALIDATOR_FREE_CLAIM:{seed}:{task_id}:{field}"
                )
    if seen != set(row_by_id):
        raise ValueError(f"A2_BUDGET_VALIDATOR_FREE_COVERAGE:{seed}")


def _free_summary_from_raw(
    tasks: list[dict[str, Any]], row_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    claims = [
        _recompute_free_claims(row_by_id[item["task_id"]], item) for item in tasks
    ]
    unsatisfied = [item for item in claims if not item["initial_goal_satisfied"]]
    return {
        "task_count": len(claims),
        "exact_plan_rate": _rate([item["exact_plan_match"] for item in claims]),
        "goal_success_rate": _rate([item["final_goal_success"] for item in claims]),
        "initially_unsatisfied_goal_success_rate": _rate(
            [item["final_goal_success"] for item in unsatisfied]
        ),
        "zero_action_rate": _rate([item["predicted_plan_length"] == 0 for item in claims]),
    }


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    row_by_id: dict[str, dict[str, Any]],
    *,
    seed: int,
    epoch: int,
) -> None:
    if checkpoint.get("epoch") != epoch:
        raise ValueError(f"A2_BUDGET_VALIDATOR_CHECKPOINT_EPOCH:{seed}:{epoch}")
    if checkpoint.get("update_count") != epoch * len(EXPECTED_TRAIN_TASK_IDS):
        raise ValueError(f"A2_BUDGET_VALIDATOR_CHECKPOINT_UPDATES:{seed}:{epoch}")
    teacher = checkpoint["teacher_forced"]
    free = checkpoint["free_running"]
    _validate_teacher_raw(teacher, row_by_id, seed=seed)
    _validate_free_raw(free, row_by_id, seed=seed)
    expected_teacher = _teacher_summary_from_raw(teacher)
    expected_free = _free_summary_from_raw(free, row_by_id)
    if checkpoint.get("teacher_forced_summary") != expected_teacher:
        raise ValueError(f"A2_BUDGET_VALIDATOR_CHECKPOINT_METRIC:{seed}:{epoch}")
    if checkpoint.get("free_running_summary") != expected_free:
        raise ValueError(f"A2_BUDGET_VALIDATOR_CHECKPOINT_FREE_METRIC:{seed}:{epoch}")


def _validate_position0_epoch_record(
    record: dict[str, Any], row_by_id: dict[str, dict[str, Any]], *, seed: int, epoch: int
) -> None:
    if record.get("epoch") != epoch:
        raise ValueError(f"A2_BUDGET_VALIDATOR_POS0_EPOCH:{seed}:{epoch}")
    if record.get("update_count") != epoch * len(EXPECTED_TRAIN_TASK_IDS):
        raise ValueError(f"A2_BUDGET_VALIDATOR_POS0_UPDATES:{seed}:{epoch}")
    tasks = record.get("tasks")
    if len(tasks) != len(EXPECTED_TRAIN_TASK_IDS):
        raise ValueError(f"A2_BUDGET_VALIDATOR_POS0_TASK_COUNT:{seed}:{epoch}")
    seen = set()
    for item in tasks:
        task_id = item.get("task_id")
        if task_id in seen or task_id not in row_by_id:
            raise ValueError(f"A2_BUDGET_VALIDATOR_POS0_TASK_ID:{seed}:{epoch}")
        seen.add(task_id)
        gold = row_by_id[task_id]["oracle_work_plan"][0][0]
        if item.get("gold_operator") != gold:
            raise ValueError(f"A2_BUDGET_VALIDATOR_POS0_GOLD:{seed}:{epoch}:{task_id}")
        correct = item.get("predicted_operator") == gold
        if item.get("operator_correct") != correct:
            raise ValueError(
                f"A2_BUDGET_VALIDATOR_POS0_RAW_CLAIM:{seed}:{epoch}:{task_id}"
            )
        probability_gold = float(item["probability_gold_operator"])
        probability_end = float(item["probability_end"])
        if not 0.0 <= probability_gold <= 1.0 or not 0.0 <= probability_end <= 1.0:
            raise ValueError(
                f"A2_BUDGET_VALIDATOR_POS0_PROBABILITY:{seed}:{epoch}:{task_id}"
            )
        expected_nll = -math.log(max(probability_gold, 1.1754943508222875e-38))
        if float(item["operator_nll"]) != expected_nll:
            raise ValueError(f"A2_BUDGET_VALIDATOR_POS0_NLL:{seed}:{epoch}:{task_id}")
    if seen != set(EXPECTED_TRAIN_TASK_IDS):
        raise ValueError(f"A2_BUDGET_VALIDATOR_POS0_COVERAGE:{seed}:{epoch}")


def _first_rescue_from_raw(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        rescued = [
            item["task_id"]
            for item in record["tasks"]
            if item["gold_operator"] == "UNSTACK" and item["operator_correct"]
        ]
        if rescued:
            return {
                "epoch": record["epoch"],
                "update_count": record["update_count"],
                "task_ids": rescued,
            }
    return None


def _validate_trajectory_claim(
    records: list[dict[str, Any]],
    claimed_first_rescue: dict[str, Any] | None,
    row_by_id: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any] | None:
    if len(records) != MAX_EPOCH:
        raise ValueError(f"A2_BUDGET_VALIDATOR_POS0_EPOCH_COUNT:{seed}")
    for epoch, record in enumerate(records, start=1):
        _validate_position0_epoch_record(record, row_by_id, seed=seed, epoch=epoch)
    expected = _first_rescue_from_raw(records)
    if claimed_first_rescue != expected:
        raise ValueError(f"A2_BUDGET_VALIDATOR_FIRST_RESCUE_CLAIM:{seed}")
    return expected


def _global_first_rescue(by_seed: dict[str, dict[str, Any] | None]) -> dict[str, Any] | None:
    candidates = []
    for seed, rescue in by_seed.items():
        if rescue is not None:
            candidates.append({"seed": int(seed), **rescue})
    if not candidates:
        return None
    epoch = min(item["epoch"] for item in candidates)
    return {"epoch": epoch, "events": [item for item in candidates if item["epoch"] == epoch]}


def _aggregate_checkpoint_from_raw(
    seed_results: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    epoch: int,
) -> dict[str, Any]:
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
    teacher_summary = _teacher_summary_from_raw(teacher)
    return {
        "epoch": epoch,
        "update_count_per_seed": epoch * len(EXPECTED_TRAIN_TASK_IDS),
        "teacher_forced": teacher_summary,
        "free_running": _free_summary_from_raw(free, row_by_id),
        "mean_within_seed_position0_task_discrimination": {
            key: _mean([float(item[key]) for item in discriminations])
            for key in discriminations[0]
        },
    }


def validate_claims_from_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = generate_train_only()
    rows = list(dataset["train"])
    rows.sort(key=lambda row: row["task_id"])
    row_by_id = {row["task_id"]: row for row in rows}
    if tuple(row_by_id) != EXPECTED_TRAIN_TASK_IDS:
        raise ValueError("A2_BUDGET_VALIDATOR_TRAIN_SPLIT")
    if set(row_by_id) & set(HELDOUT):
        raise ValueError("A2_BUDGET_VALIDATOR_HELDOUT_ACCESS")
    if payload.get("seeds") != list(SEEDS):
        raise ValueError("A2_BUDGET_VALIDATOR_SEEDS")
    if payload.get("checkpoint_epochs") != list(CHECKPOINT_EPOCHS):
        raise ValueError("A2_BUDGET_VALIDATOR_CHECKPOINTS")
    if payload.get("max_epoch") != MAX_EPOCH:
        raise ValueError("A2_BUDGET_VALIDATOR_MAX_EPOCH")
    if payload.get("canonical_task_order") != list(EXPECTED_TRAIN_TASK_IDS):
        raise ValueError("A2_BUDGET_VALIDATOR_TASK_ORDER")
    if payload.get("heldout_accessed") is not False:
        raise ValueError("A2_BUDGET_VALIDATOR_HELDOUT_FLAG")
    if payload.get("dataset", {}).get("frozen_dataset_lineage_hash") != (
        FROZEN_DATASET_LINEAGE_HASH_V1
    ):
        raise ValueError("A2_BUDGET_VALIDATOR_DATASET_LINEAGE")
    if payload.get("dataset", {}).get("evaluated_task_ids") != list(
        EXPECTED_TRAIN_TASK_IDS
    ):
        raise ValueError("A2_BUDGET_VALIDATOR_TASK_COVERAGE")
    seed_results = payload.get("seed_results")
    if len(seed_results) != len(SEEDS) or {item["seed"] for item in seed_results} != set(SEEDS):
        raise ValueError("A2_BUDGET_VALIDATOR_SEED_RESULTS")
    rescues: dict[str, dict[str, Any] | None] = {}
    for seed in SEEDS:
        result = next(item for item in seed_results if item["seed"] == seed)
        _validate_training_schedule(result, row_by_id, seed=seed)
        _validate_prefix_equivalence(result["prefix_equivalence"], result, seed=seed)
        checkpoints = result["checkpoints"]
        if [item["epoch"] for item in checkpoints] != list(CHECKPOINT_EPOCHS):
            raise ValueError(f"A2_BUDGET_VALIDATOR_CHECKPOINT_COVERAGE:{seed}")
        for epoch, checkpoint in zip(CHECKPOINT_EPOCHS, checkpoints, strict=True):
            _validate_checkpoint(checkpoint, row_by_id, seed=seed, epoch=epoch)
        rescues[str(seed)] = _validate_trajectory_claim(
            result["position0_epoch_evidence"],
            result.get("first_position0_unstack_rescue"),
            row_by_id,
            seed=seed,
        )
    expected_by_seed = rescues
    expected_global = _global_first_rescue(rescues)
    claims = payload.get("trajectory_claims")
    if claims.get("first_position0_unstack_rescue_by_seed") != expected_by_seed:
        raise ValueError("A2_BUDGET_VALIDATOR_TRAJECTORY_BY_SEED")
    if claims.get("first_position0_unstack_rescue_global") != expected_global:
        raise ValueError("A2_BUDGET_VALIDATOR_TRAJECTORY_GLOBAL")
    expected_aggregates = {
        str(epoch): _aggregate_checkpoint_from_raw(seed_results, row_by_id, epoch=epoch)
        for epoch in CHECKPOINT_EPOCHS
    }
    if payload.get("checkpoint_aggregates") != expected_aggregates:
        raise ValueError("A2_BUDGET_VALIDATOR_CHECKPOINT_AGGREGATES")
    return {
        "prefix_equivalence": "PASS",
        "claim_validation": "INDEPENDENT_FROM_PERSISTED_RAW_EVIDENCE",
        "trajectory_claim_validation": "INDEPENDENT_FIRST_RESCUE_RECOMPUTATION",
    }
