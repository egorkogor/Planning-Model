"""Independent evidence validator for the A2 causal microexperiment.

This module intentionally does not call the producer's training, aggregation, metric,
weighting, or contrast helpers. It recomputes claim-bearing values from persisted raw
evidence and the frozen train-only materialization.
"""

from __future__ import annotations

from typing import Any

from .dataset import task_from_row
from .domain import apply_action, goal_satisfied, validate_state
from .e2e import parse_nonterminal_step
from .train_only_dataset import FROZEN_DATASET_LINEAGE_HASH_V1, generate_train_only

ARM_CONTROL = "canonical_control"
ARM_EQUAL_POSITION = "equal_position_operator_loss"
ARM_ORDER_ONLY = "task_order_only"
ARMS = (ARM_CONTROL, ARM_EQUAL_POSITION, ARM_ORDER_ONLY)
CANONICAL_ORDER = ("bw-00000001", "bw-00000002", "bw-00000003")
ORDER_ONLY_ORDER = ("bw-00000002", "bw-00000003", "bw-00000001")
HELDOUT = ("bw-00000004", "bw-00000005")
EPOCHS = 3
EQUIVALENCE_TRACE_FIELDS = (
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

EXPECTED_ARM_CONTRACT = {
    ARM_CONTROL: {
        "trainer": (
            "planner_toy.learnability._train_a2_with_loss_trace -> "
            "planner_toy.quality._train"
        ),
        "operator_loss": "per-task-mean-cross-entropy",
        "task_order": list(CANONICAL_ORDER),
        "changed_dimension": "NONE",
    },
    ARM_EQUAL_POSITION: {
        "trainer": "microexperiment-intervention",
        "operator_loss": "equal-per-position-cross-entropy-epoch-scale-preserved",
        "task_order": list(CANONICAL_ORDER),
        "changed_dimension": "OPERATOR_LOSS_NORMALIZATION_ONLY",
    },
    ARM_ORDER_ONLY: {
        "trainer": "microexperiment-intervention",
        "operator_loss": "per-task-mean-cross-entropy",
        "task_order": list(ORDER_ONLY_ORDER),
        "changed_dimension": "TASK_ORDER_ONLY",
    },
}


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _validate_teacher_raw(
    tasks: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> None:
    if len(tasks) != len(row_by_id):
        raise ValueError(f"A2_CAUSAL_VALIDATOR_TEACHER_TASK_COUNT:{seed}")
    seen: set[str] = set()
    for task in tasks:
        task_id = task["task_id"]
        if task_id in seen or task_id not in row_by_id:
            raise ValueError(f"A2_CAUSAL_VALIDATOR_TEACHER_TASK_ID:{seed}")
        seen.add(task_id)
        if task.get("seed") != seed or task.get("split") != "train":
            raise ValueError(f"A2_CAUSAL_VALIDATOR_TEACHER_SCOPE:{seed}:{task_id}")
        row = row_by_id[task_id]
        positions = task["positions"]
        if len(positions) != len(row["oracle_work_plan"]):
            raise ValueError(f"A2_CAUSAL_VALIDATOR_TEACHER_POSITION_COUNT:{seed}:{task_id}")
        for index, (position, step) in enumerate(
            zip(positions, row["oracle_work_plan"], strict=True)
        ):
            gold_operator = step[0]
            gold_arg1 = step[1] if len(step) > 1 else None
            gold_arg2 = step[2] if len(step) > 2 else None
            if position.get("position_index") != index:
                raise ValueError(
                    f"A2_CAUSAL_VALIDATOR_TEACHER_POSITION_INDEX:{seed}:{task_id}:{index}"
                )
            if (
                position.get("gold_operator") != gold_operator
                or position.get("gold_arg1") != gold_arg1
                or position.get("gold_arg2") != gold_arg2
            ):
                raise ValueError(
                    f"A2_CAUSAL_VALIDATOR_TEACHER_GOLD:{seed}:{task_id}:{index}"
                )
            expected_arg1_target = gold_arg1 is not None
            expected_arg2_target = gold_arg2 is not None
            if (
                position.get("has_arg1_target") != expected_arg1_target
                or position.get("has_arg2_target") != expected_arg2_target
            ):
                raise ValueError(
                    f"A2_CAUSAL_VALIDATOR_TEACHER_TARGET_FLAGS:{seed}:{task_id}:{index}"
                )
            operator_correct = position["predicted_operator"] == gold_operator
            arg1_correct = (
                position["arg1_head_prediction"] == gold_arg1
                if expected_arg1_target
                else None
            )
            arg2_correct = (
                position["arg2_head_prediction"] == gold_arg2
                if expected_arg2_target
                else None
            )
            joint = (
                operator_correct
                and arg1_correct is not False
                and arg2_correct is not False
            )
            if position.get("operator_correct") != operator_correct:
                raise ValueError(
                    f"A2_CAUSAL_VALIDATOR_TEACHER_OPERATOR_CLAIM:{seed}:{task_id}:{index}"
                )
            if position.get("arg1_correct") != arg1_correct:
                raise ValueError(
                    f"A2_CAUSAL_VALIDATOR_TEACHER_ARG1_CLAIM:{seed}:{task_id}:{index}"
                )
            if position.get("arg2_correct") != arg2_correct:
                raise ValueError(
                    f"A2_CAUSAL_VALIDATOR_TEACHER_ARG2_CLAIM:{seed}:{task_id}:{index}"
                )
            if position.get("joint_step_correct") != joint:
                raise ValueError(
                    f"A2_CAUSAL_VALIDATOR_TEACHER_JOINT_CLAIM:{seed}:{task_id}:{index}"
                )
            probability_end = float(position["probability_end"])
            if not 0.0 <= probability_end <= 1.0:
                raise ValueError(
                    f"A2_CAUSAL_VALIDATOR_TEACHER_END_PROBABILITY:{seed}:{task_id}:{index}"
                )
    if seen != set(row_by_id):
        raise ValueError(f"A2_CAUSAL_VALIDATOR_TEACHER_COVERAGE:{seed}")


def _teacher_metrics_from_raw(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    positions: list[dict[str, Any]] = []
    for task in tasks:
        positions.extend(task["positions"])
    end = [item for item in positions if item["gold_operator"] == "END"]
    non_end = [item for item in positions if item["gold_operator"] != "END"]
    arg1 = [item for item in positions if item["has_arg1_target"]]
    arg2 = [item for item in positions if item["has_arg2_target"]]
    pos0 = [
        item
        for item in positions
        if item["position_index"] == 0 and item["gold_operator"] == "UNSTACK"
    ]
    pos4 = [
        item
        for item in positions
        if item["position_index"] == 4 and item["gold_operator"] == "END"
    ]

    def operator_correct(item: dict[str, Any]) -> bool:
        return item["predicted_operator"] == item["gold_operator"]

    def arg1_correct(item: dict[str, Any]) -> bool:
        return item["arg1_head_prediction"] == item["gold_arg1"]

    def arg2_correct(item: dict[str, Any]) -> bool:
        return item["arg2_head_prediction"] == item["gold_arg2"]

    def joint_correct(item: dict[str, Any]) -> bool:
        return (
            operator_correct(item)
            and (not item["has_arg1_target"] or arg1_correct(item))
            and (not item["has_arg2_target"] or arg2_correct(item))
        )

    return {
        "operator_accuracy": _rate([operator_correct(item) for item in positions]),
        "non_end_operator_accuracy": _rate([operator_correct(item) for item in non_end]),
        "end_accuracy": _rate([operator_correct(item) for item in end]),
        "arg1_accuracy": _rate([arg1_correct(item) for item in arg1]),
        "arg2_accuracy": _rate([arg2_correct(item) for item in arg2]),
        "joint_step_accuracy": _rate([joint_correct(item) for item in positions]),
        "predicted_end_rate": _rate(
            [item["predicted_operator"] == "END" for item in positions]
        ),
        "position0_unstack": {
            "target_count": len(pos0),
            "accuracy": _rate([operator_correct(item) for item in pos0]),
            "mean_end_probability": _mean(
                [float(item["probability_end"]) for item in pos0]
            ),
        },
        "position4_end": {
            "target_count": len(pos4),
            "accuracy": _rate([operator_correct(item) for item in pos4]),
            "mean_end_probability": _mean(
                [float(item["probability_end"]) for item in pos4]
            ),
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
    full_plan_executable = (
        terminal_end
        and executable
        and (predicted_plan_length > 0 or initial_goal_satisfied)
    )
    final_goal_success = full_plan_executable and goal_satisfied(state, task.goal)
    return {
        "initial_goal_satisfied": initial_goal_satisfied,
        "predicted_plan_length": predicted_plan_length,
        "exact_plan_match": raw == row["oracle_work_plan"],
        "final_goal_success": final_goal_success,
    }


def _validate_free_raw(
    tasks: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    if len(tasks) != len(row_by_id):
        raise ValueError(f"A2_CAUSAL_VALIDATOR_FREE_TASK_COUNT:{seed}")
    claims: dict[str, dict[str, Any]] = {}
    for item in tasks:
        task_id = item["task_id"]
        if task_id in claims or task_id not in row_by_id:
            raise ValueError(f"A2_CAUSAL_VALIDATOR_FREE_TASK_ID:{seed}")
        if item.get("seed") != seed or item.get("split") != "train":
            raise ValueError(f"A2_CAUSAL_VALIDATOR_FREE_SCOPE:{seed}:{task_id}")
        recomputed = _recompute_free_claims(row_by_id[task_id], item)
        for field, expected in recomputed.items():
            if item.get(field) != expected:
                raise ValueError(
                    f"A2_CAUSAL_VALIDATOR_FREE_CLAIM:{seed}:{task_id}:{field}"
                )
        claims[task_id] = recomputed
    if set(claims) != set(row_by_id):
        raise ValueError(f"A2_CAUSAL_VALIDATOR_FREE_COVERAGE:{seed}")
    return claims


def _free_metrics_from_raw(
    tasks: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    claims = {
        item["task_id"]: _recompute_free_claims(row_by_id[item["task_id"]], item)
        for item in tasks
    }
    unsatisfied = [value for value in claims.values() if not value["initial_goal_satisfied"]]
    return {
        "task_count": len(tasks),
        "exact_plan_rate": _rate([value["exact_plan_match"] for value in claims.values()]),
        "goal_success_rate": _rate([value["final_goal_success"] for value in claims.values()]),
        "zero_action_rate": _rate(
            [value["predicted_plan_length"] == 0 for value in claims.values()]
        ),
        "initially_unsatisfied_goal_success_rate": _rate(
            [value["final_goal_success"] for value in unsatisfied]
        ),
    }


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _contrast_from_aggregates(
    intervention: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    it = intervention["teacher_forced"]
    ct = control["teacher_forced"]
    iff = intervention["free_running"]
    cff = control["free_running"]
    return {
        "position0_unstack_accuracy_delta": _delta(
            it["position0_unstack"]["accuracy"], ct["position0_unstack"]["accuracy"]
        ),
        "position0_unstack_end_probability_delta": _delta(
            it["position0_unstack"]["mean_end_probability"],
            ct["position0_unstack"]["mean_end_probability"],
        ),
        "position4_end_accuracy_delta": _delta(
            it["position4_end"]["accuracy"], ct["position4_end"]["accuracy"]
        ),
        "operator_accuracy_delta": _delta(it["operator_accuracy"], ct["operator_accuracy"]),
        "joint_step_accuracy_delta": _delta(
            it["joint_step_accuracy"], ct["joint_step_accuracy"]
        ),
        "free_running_goal_success_rate_delta": _delta(
            iff["goal_success_rate"], cff["goal_success_rate"]
        ),
        "free_running_initially_unsatisfied_goal_success_rate_delta": _delta(
            iff["initially_unsatisfied_goal_success_rate"],
            cff["initially_unsatisfied_goal_success_rate"],
        ),
    }


def _expected_weighting_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = {row["task_id"]: len(row["oracle_work_plan"]) for row in rows}
    total = sum(lengths.values())
    task_count = len(rows)
    equal = task_count / total
    end_weight = 1.0 / lengths["bw-00000001"]
    unstack_weight = (
        1.0 / lengths["bw-00000002"] + 1.0 / lengths["bw-00000003"]
    )
    return {
        "operator_position_count_per_epoch": total,
        "task_count": task_count,
        "valid_operator_positions_by_task": lengths,
        "canonical_position0_end_weight_per_epoch": end_weight,
        "canonical_position0_unstack_weight_per_epoch": unstack_weight,
        "canonical_end_to_unstack_position0_weight_ratio": end_weight / unstack_weight,
        "equal_position_weight_each": equal,
        "equal_position_position0_end_weight_per_epoch": equal,
        "equal_position_position0_unstack_weight_per_epoch": 2 * equal,
        "equal_position_end_to_unstack_position0_weight_ratio": 0.5,
        "epoch_scale_preservation": {
            "canonical_uniform_loss_operator_weight_sum": float(task_count),
            "equal_position_uniform_loss_operator_weight_sum": equal * total,
        },
    }


def _validate_training_schedule(
    arm: str,
    training: dict[str, Any],
    row_lengths: dict[str, int],
) -> None:
    order = CANONICAL_ORDER if arm != ARM_ORDER_ONLY else ORDER_ONLY_ORDER
    expected_tasks = list(order) * EPOCHS
    updates = training["updates"]
    if len(updates) != len(expected_tasks):
        raise ValueError(f"A2_CAUSAL_VALIDATOR_UPDATE_COUNT:{arm}")
    total_positions = sum(row_lengths.values())
    task_count = len(row_lengths)
    for index, (update, task_id) in enumerate(zip(updates, expected_tasks, strict=True)):
        if update["update_index"] != index:
            raise ValueError(f"A2_CAUSAL_VALIDATOR_UPDATE_INDEX:{arm}:{index}")
        if update["epoch_index"] != index // len(order):
            raise ValueError(f"A2_CAUSAL_VALIDATOR_EPOCH_INDEX:{arm}:{index}")
        if update["task_id"] != task_id:
            raise ValueError(f"A2_CAUSAL_VALIDATOR_TASK_ORDER:{arm}:{index}")
        target_count = row_lengths[task_id]
        if update["operator_target_count"] != target_count:
            raise ValueError(f"A2_CAUSAL_VALIDATOR_OPERATOR_TARGET_COUNT:{arm}:{index}")
        if arm == ARM_EQUAL_POSITION:
            expected_weight = task_count / total_positions
        else:
            expected_weight = 1.0 / target_count
        if update["operator_position_weight"] != expected_weight:
            raise ValueError(f"A2_CAUSAL_VALIDATOR_POSITION_WEIGHT:{arm}:{index}")
        if update["gradient_clip_norm"] != 1.0:
            raise ValueError(f"A2_CAUSAL_VALIDATOR_CLIP_NORM:{arm}:{index}")


def _validate_equivalence(
    payload: dict[str, Any], result_map: dict[tuple[int, str], dict[str, Any]]
) -> None:
    records = payload.get("control_equivalence")
    seeds = payload["seeds"]
    if not isinstance(records, list) or len(records) != len(seeds):
        raise ValueError("A2_CAUSAL_VALIDATOR_EQUIVALENCE_COVERAGE")
    by_seed = {record["seed"]: record for record in records}
    if set(by_seed) != set(seeds):
        raise ValueError("A2_CAUSAL_VALIDATOR_EQUIVALENCE_SEEDS")
    for seed in seeds:
        record = by_seed[seed]
        if record.get("status") != "PASS":
            raise ValueError(f"A2_CAUSAL_VALIDATOR_EQUIVALENCE_STATUS:{seed}")
        if record.get("purpose") != "NON_SCIENTIFIC_INTERVENTION_TRAINER_CONTROL_EQUIVALENCE":
            raise ValueError(f"A2_CAUSAL_VALIDATOR_EQUIVALENCE_PURPOSE:{seed}")
        if record.get("task_order") != list(CANONICAL_ORDER):
            raise ValueError(f"A2_CAUSAL_VALIDATOR_EQUIVALENCE_ORDER:{seed}")
        if record.get("operator_loss") != "per-task-mean-cross-entropy":
            raise ValueError(f"A2_CAUSAL_VALIDATOR_EQUIVALENCE_LOSS:{seed}")
        if record.get("trace_fields") != list(EQUIVALENCE_TRACE_FIELDS):
            raise ValueError(f"A2_CAUSAL_VALIDATOR_EQUIVALENCE_FIELDS:{seed}")
        actual_control = result_map[(seed, ARM_CONTROL)]["training"]
        projected_control = {
            "initialization_canonical_sha256": actual_control[
                "initialization_canonical_sha256"
            ],
            "trained_canonical_sha256": actual_control["trained_canonical_sha256"],
            "optimizer_canonical_sha256": actual_control["optimizer_canonical_sha256"],
            "updates": [
                {field: update[field] for field in EQUIVALENCE_TRACE_FIELDS}
                for update in actual_control["updates"]
            ],
        }
        if record["control"] != projected_control:
            raise ValueError(f"A2_CAUSAL_VALIDATOR_EQUIVALENCE_CONTROL_BINDING:{seed}")
        if record["probe"] != record["control"]:
            raise ValueError(f"A2_CAUSAL_VALIDATOR_EQUIVALENCE_EXACT_MISMATCH:{seed}")


def validate_claims_from_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate all claim-bearing summaries from persisted raw evidence only."""
    dataset = generate_train_only()
    rows = list(dataset["train"])
    row_ids = [row["task_id"] for row in rows]
    if row_ids != ["bw-00000001", "bw-00000003", "bw-00000002"]:
        raise ValueError("A2_CAUSAL_VALIDATOR_LINEAGE_ORDER")
    if set(row_ids) & set(HELDOUT):
        raise ValueError("A2_CAUSAL_VALIDATOR_HELDOUT")
    if payload.get("arm_contract") != EXPECTED_ARM_CONTRACT:
        raise ValueError("A2_CAUSAL_VALIDATOR_ARM_CONTRACT")
    if payload.get("arms") != list(ARMS):
        raise ValueError("A2_CAUSAL_VALIDATOR_ARMS")
    expected_dataset = {
        "schema_version": dataset["schema_version"],
        "frozen_dataset_lineage_hash": FROZEN_DATASET_LINEAGE_HASH_V1,
        "evaluated_train_split_hash": dataset["evaluated_train_split_hash"],
        "dataset_lineage_order": dataset["train_task_ids"],
        "evaluated_task_ids": list(CANONICAL_ORDER),
    }
    if payload.get("dataset") != expected_dataset:
        raise ValueError("A2_CAUSAL_VALIDATOR_DATASET")
    if payload.get("weighting_contract") != _expected_weighting_contract(rows):
        raise ValueError("A2_CAUSAL_VALIDATOR_WEIGHTING_CONTRACT")

    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("A2_CAUSAL_VALIDATOR_SEEDS")
    seed_results = payload.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) != len(seeds) * len(ARMS):
        raise ValueError("A2_CAUSAL_VALIDATOR_SEED_RESULT_COVERAGE")
    result_map: dict[tuple[int, str], dict[str, Any]] = {}
    row_by_id = {row["task_id"]: row for row in rows}
    row_lengths = {row["task_id"]: len(row["oracle_work_plan"]) for row in rows}
    for result in seed_results:
        key = (result["seed"], result["arm"])
        if key in result_map or result["seed"] not in seeds or result["arm"] not in ARMS:
            raise ValueError("A2_CAUSAL_VALIDATOR_SEED_RESULT_KEY")
        if result["contract"] != EXPECTED_ARM_CONTRACT[result["arm"]]:
            raise ValueError(f"A2_CAUSAL_VALIDATOR_RESULT_CONTRACT:{result['arm']}")
        _validate_training_schedule(result["arm"], result["training"], row_lengths)
        _validate_teacher_raw(result["teacher_forced"], row_by_id, seed=result["seed"])
        expected_teacher = _teacher_metrics_from_raw(result["teacher_forced"])
        if result["teacher_forced_metrics"] != expected_teacher:
            raise ValueError(
                f"A2_CAUSAL_VALIDATOR_SEED_TEACHER_METRICS:{result['seed']}:{result['arm']}"
            )
        _validate_free_raw(result["free_running"], row_by_id, seed=result["seed"])
        expected_free = _free_metrics_from_raw(result["free_running"], row_by_id)
        if result["free_running_metrics"] != expected_free:
            raise ValueError(
                f"A2_CAUSAL_VALIDATOR_SEED_FREE_METRICS:{result['seed']}:{result['arm']}"
            )
        result_map[key] = result
    if set(result_map) != {(seed, arm) for seed in seeds for arm in ARMS}:
        raise ValueError("A2_CAUSAL_VALIDATOR_SEED_RESULT_MATRIX")

    _validate_equivalence(payload, result_map)

    recomputed_aggregates: dict[str, Any] = {}
    for arm in ARMS:
        teacher_tasks: list[dict[str, Any]] = []
        free_tasks: list[dict[str, Any]] = []
        for seed in seeds:
            result = result_map[(seed, arm)]
            teacher_tasks.extend(result["teacher_forced"])
            free_tasks.extend(result["free_running"])
        recomputed_aggregates[arm] = {
            "seed_count": len(seeds),
            "teacher_forced": _teacher_metrics_from_raw(teacher_tasks),
            "free_running": _free_metrics_from_raw(free_tasks, row_by_id),
        }
    if payload.get("aggregates") != recomputed_aggregates:
        raise ValueError("A2_CAUSAL_VALIDATOR_AGGREGATES")

    control = recomputed_aggregates[ARM_CONTROL]
    expected_contrasts = {
        ARM_EQUAL_POSITION: _contrast_from_aggregates(
            recomputed_aggregates[ARM_EQUAL_POSITION], control
        ),
        ARM_ORDER_ONLY: _contrast_from_aggregates(
            recomputed_aggregates[ARM_ORDER_ONLY], control
        ),
    }
    if payload.get("contrasts_vs_control") != expected_contrasts:
        raise ValueError("A2_CAUSAL_VALIDATOR_CONTRASTS")
    if payload.get("hypothesis", {}).get("causal_result") is not None:
        raise ValueError("A2_CAUSAL_VALIDATOR_CAUSAL_RESULT")
    if payload.get("interpretation_policy") != {
        "label": "SUPPORTED HYPOTHESIS / NOT PROVEN",
        "automatic_gate": None,
        "scientific_status": "REDESIGN",
    }:
        raise ValueError("A2_CAUSAL_VALIDATOR_INTERPRETATION_POLICY")
    return {
        "control_equivalence": "PASS",
        "claim_validation": "INDEPENDENT_FROM_PERSISTED_RAW_EVIDENCE",
    }
