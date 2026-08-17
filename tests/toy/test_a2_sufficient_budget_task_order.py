from __future__ import annotations

import copy
import math
import subprocess

import pytest
import torch

import planner_toy.a2_sufficient_budget_task_order_validator as validator
from planner_toy.a2_sufficient_budget_task_order import (
    ARMS,
    CHECKPOINT_EPOCHS,
    _arm_summaries,
    _cross_arm_deltas,
)
from planner_toy.a2_sufficient_budget_task_order_validator import (
    MAX_EPOCH,
    SEEDS,
    TASKS,
    _free_summary,
    _recompute_arm_summaries,
    _recompute_deltas,
    _source_identity_at_commit,
    _teacher_summary,
    validate_claims_from_evidence,
)
from planner_toy.canonical import sha256
from planner_toy.train_only_dataset import generate_train_only

RESCUE = {
    "canonical_order": (11, 12),
    "task01_middle": (10, 11),
    "task01_last": (13, 14),
}


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


def _position(row, index: int, *, p0_solved: bool) -> dict:
    step = row["oracle_work_plan"][index]
    gold = step[0]
    predicted = gold
    if index == 0 and gold == "UNSTACK" and not p0_solved:
        predicted = "END"
    p_gold = 0.9 if predicted == gold else 0.1
    p_end = 0.9 if predicted == "END" else 0.1
    gold_arg1 = step[1] if len(step) > 1 else None
    gold_arg2 = step[2] if len(step) > 2 else None
    arg1_correct = True if gold_arg1 is not None else None
    arg2_correct = True if gold_arg2 is not None else None
    operator_correct = predicted == gold
    return {
        "split": "train",
        "task_id": row["task_id"],
        "seed": 0,
        "position_index": index,
        "gold_operator": gold,
        "predicted_operator": predicted,
        "operator_correct": operator_correct,
        "end_correct": operator_correct if gold == "END" else None,
        "probability_gold_operator": p_gold,
        "probability_end": p_end,
        "operator_nll": -math.log(max(p_gold, torch.finfo(torch.float32).tiny)),
        "has_arg1_target": gold_arg1 is not None,
        "has_arg2_target": gold_arg2 is not None,
        "gold_arg1": gold_arg1,
        "gold_arg2": gold_arg2,
        "arg1_head_prediction": gold_arg1,
        "arg2_head_prediction": gold_arg2,
        "decoded_arg1": gold_arg1 if predicted != "END" else None,
        "decoded_arg2": gold_arg2 if predicted in {"UNSTACK", "STACK"} else None,
        "arg1_correct": arg1_correct,
        "arg2_correct": arg2_correct,
        "joint_step_correct": (
            operator_correct and arg1_correct is not False and arg2_correct is not False
        ),
    }


def _teacher_task(row, seed: int, *, p0_solved: bool) -> dict:
    positions = [
        _position(row, index, p0_solved=p0_solved)
        for index in range(len(row["oracle_work_plan"]))
    ]
    for position in positions:
        position["seed"] = seed
    return {
        "split": "train",
        "task_id": row["task_id"],
        "seed": seed,
        "positions": positions,
    }


def _free_task(row, seed: int, *, solved: bool) -> dict:
    plan = copy.deepcopy(row["oracle_work_plan"]) if solved else [["END"]]
    initial = row["task_id"] == "bw-00000001"
    return {
        "split": "train",
        "task_id": row["task_id"],
        "seed": seed,
        "predicted_plan": plan,
        "predicted_plan_length": max(len(plan) - 1, 0),
        "exact_plan_match": plan == row["oracle_work_plan"],
        "initial_goal_satisfied": initial,
        "final_goal_success": solved or initial,
    }


def _epoch_record(rows, seed: int, epoch: int, *, p0_epoch: int, free_epoch: int) -> dict:
    p0 = []
    free = []
    for row in rows:
        nontrivial = row["task_id"] in {"bw-00000002", "bw-00000003"}
        teacher = _teacher_task(
            row,
            seed,
            p0_solved=(not nontrivial) or epoch >= p0_epoch,
        )
        first = teacher["positions"][0]
        p0.append(
            {
                "task_id": row["task_id"],
                "gold_operator": first["gold_operator"],
                "predicted_operator": first["predicted_operator"],
                "operator_correct": first["operator_correct"],
                "probability_gold_operator": first["probability_gold_operator"],
                "operator_nll": first["operator_nll"],
                "probability_end": first["probability_end"],
            }
        )
        task = _free_task(
            row,
            seed,
            solved=(not nontrivial) or epoch >= free_epoch,
        )
        free.append(
            {
                "task_id": task["task_id"],
                "initial_goal_satisfied": task["initial_goal_satisfied"],
                "predicted_plan": task["predicted_plan"],
                "predicted_plan_length": task["predicted_plan_length"],
                "exact_plan_match": task["exact_plan_match"],
                "final_goal_success": task["final_goal_success"],
                "failure_code": None if task["final_goal_success"] else "GOAL_NOT_ACHIEVED",
            }
        )
    return {
        "epoch": epoch,
        "update_count": epoch * 3,
        "position0": p0,
        "free_running": free,
    }


def _updates(rows, arm: str) -> list[dict]:
    row_by_id = {row["task_id"]: row for row in rows}
    records = []
    for epoch_index in range(MAX_EPOCH):
        for task_id in ARMS[arm]:
            row = row_by_id[task_id]
            operator = len(row["oracle_work_plan"])
            arg1 = sum(step[0] != "END" for step in row["oracle_work_plan"])
            arg2 = sum(step[0] in {"UNSTACK", "STACK"} for step in row["oracle_work_plan"])
            arg1_loss = 1.0 if arg1 else None
            arg2_loss = 1.0 if arg2 else None
            records.append(
                {
                    "update_index": len(records),
                    "epoch_index": epoch_index,
                    "task_id": task_id,
                    "operator_loss": 1.0,
                    "arg1_pointer_loss": arg1_loss,
                    "arg2_pointer_loss": arg2_loss,
                    "total_loss": validator._float32_total(1.0, arg1_loss, arg2_loss),
                    "operator_target_count": operator,
                    "arg1_target_count": arg1,
                    "arg2_target_count": arg2,
                    "gradient_norm": 2.0,
                    "gradient_clip_norm": 1.0,
                    "clipping_occurred": True,
                    "operator_position_weight": 1.0 / operator,
                }
            )
    return records


def _result(rows, arm: str, seed: int) -> dict:
    p0_epoch, free_epoch = RESCUE[arm]
    epoch_evidence = [
        _epoch_record(rows, seed, epoch, p0_epoch=p0_epoch, free_epoch=free_epoch)
        for epoch in range(1, MAX_EPOCH + 1)
    ]
    checkpoints = []
    for epoch in CHECKPOINT_EPOCHS:
        teachers = []
        frees = []
        for row in rows:
            nontrivial = row["task_id"] in {"bw-00000002", "bw-00000003"}
            teachers.append(
                _teacher_task(
                    row,
                    seed,
                    p0_solved=(not nontrivial) or epoch >= p0_epoch,
                )
            )
            frees.append(
                _free_task(
                    row,
                    seed,
                    solved=(not nontrivial) or epoch >= free_epoch,
                )
            )
        checkpoints.append(
            {
                "epoch": epoch,
                "update_count": epoch * 3,
                "trained_canonical_sha256": f"sha256:trained:{arm}:{seed}:{epoch}",
                "optimizer_canonical_sha256": f"sha256:optimizer:{arm}:{seed}:{epoch}",
                "teacher_forced": teachers,
                "teacher_forced_summary": _teacher_summary(teachers),
                "free_running": frees,
                "free_running_summary": _free_summary(frees),
            }
        )
    updates = _updates(rows, arm)
    p0_event = {"epoch": p0_epoch, "update_count": p0_epoch * 3}
    free_event = {"epoch": free_epoch, "update_count": free_epoch * 3}
    result = {
        "arm": arm,
        "task_order": list(ARMS[arm]),
        "seed": seed,
        "initialization_canonical_sha256": f"sha256:init:{seed}",
        "final_trained_canonical_sha256": f"sha256:final:{arm}:{seed}",
        "final_optimizer_canonical_sha256": f"sha256:finalopt:{arm}:{seed}",
        "updates": updates,
        "checkpoints": checkpoints,
        "epoch_evidence": epoch_evidence,
        "rescue_events": {
            "first_position0_operator_rescue": p0_event,
            "first_full_free_running_rescue": free_event,
        },
        "rescue_persistence": {
            "position0_operator_rescue": {
                str(epoch): (None if epoch < p0_epoch else True)
                for epoch in (10, 30, 100)
            },
            "full_free_running_rescue": {
                str(epoch): (None if epoch < free_epoch else True)
                for epoch in (10, 30, 100)
            },
        },
        "prefix_equivalence": None,
    }
    if arm == "canonical_order":
        actual = validator._actual_prefix(result)
        result["prefix_equivalence"] = {
            "seed": seed,
            "status": "PASS",
            "purpose": "NON_SCIENTIFIC_FROZEN_3_EPOCH_PREFIX_EQUIVALENCE",
            "trace_fields": list(validator.PREFIX_TRACE_FIELDS),
            "control": copy.deepcopy(actual),
            "arm_prefix": copy.deepcopy(actual),
        }
    return result


def _payload() -> tuple[dict, str]:
    commit = _commit()
    dataset = generate_train_only()
    rows = sorted(list(dataset["train"]), key=lambda row: row["task_id"])
    results = [_result(rows, arm, seed) for seed in SEEDS for arm in ARMS]
    source = _source_identity_at_commit(commit)
    payload = {
        "schema_version": "development-a2-sufficient-budget-task-order/0.1",
        "status": "development-only-scientific-microexperiment",
        "implementation_commit": commit,
        **source,
        "variant": "A2",
        "seeds": list(SEEDS),
        "arms": {name: list(order) for name, order in ARMS.items()},
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "max_epoch": MAX_EPOCH,
        "heldout_accessed": False,
        "go_latent": "NOT EVALUATED",
        "dataset": {
            "evaluated_task_ids": list(TASKS),
            "frozen_dataset_lineage_hash": dataset["frozen_dataset_lineage_hash"],
        },
        "arm_seed_results": results,
        "cross_seed_arm_summaries": _recompute_arm_summaries(results),
        "cross_arm_rescue_deltas": _recompute_deltas(results),
    }
    payload["canonical_identity"] = sha256(payload)
    return payload, commit


def _resign(payload: dict) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "canonical_identity"}
    payload["canonical_identity"] = sha256(unsigned)


@pytest.fixture(autouse=True)
def _synthetic_frozen_anchor(monkeypatch):
    original = validator._frozen_control_projection

    def fake(rows, *, seed: int, dataset_hash: str):
        del dataset_hash
        return validator._actual_prefix(_result(rows, "canonical_order", seed))

    monkeypatch.setattr(validator, "_frozen_control_projection", fake)
    return original


def test_valid_synthetic_evidence_passes() -> None:
    payload, commit = _payload()
    result = validate_claims_from_evidence(payload, implementation_commit=commit)
    assert result["independent_claim_validation"] == "PASS"
    assert result["frozen_control_anchor"] == "PASS"
    assert result["loss_decomposition_validation"] == "PASS"


def test_producer_and_validator_cross_arm_reductions_agree() -> None:
    payload, _ = _payload()
    results = payload["arm_seed_results"]
    assert _arm_summaries(results) == _recompute_arm_summaries(results)
    assert _cross_arm_deltas(results) == _recompute_deltas(results)


def test_tamper_position0_raw_with_stale_rescue_is_rejected() -> None:
    payload, commit = _payload()
    result = next(
        item
        for item in payload["arm_seed_results"]
        if item["arm"] == "canonical_order" and item["seed"] == 17
    )
    item = next(
        x
        for x in result["epoch_evidence"][10]["position0"]
        if x["task_id"] == "bw-00000002"
    )
    item.update(
        {
            "predicted_operator": "END",
            "operator_correct": False,
            "probability_gold_operator": 0.1,
            "probability_end": 0.9,
            "operator_nll": -math.log(0.1),
        }
    )
    _resign(payload)
    with pytest.raises(ValueError, match="RESCUE_EVENT"):
        validate_claims_from_evidence(payload, implementation_commit=commit)


def test_tamper_free_plan_with_stale_rescue_is_rejected() -> None:
    payload, commit = _payload()
    result = next(
        item
        for item in payload["arm_seed_results"]
        if item["arm"] == "canonical_order" and item["seed"] == 17
    )
    item = next(
        x
        for x in result["epoch_evidence"][11]["free_running"]
        if x["task_id"] == "bw-00000002"
    )
    item.update(
        {
            "predicted_plan": [["END"]],
            "predicted_plan_length": 0,
            "exact_plan_match": False,
            "final_goal_success": False,
            "failure_code": "GOAL_NOT_ACHIEVED",
        }
    )
    _resign(payload)
    with pytest.raises(ValueError, match="RESCUE_EVENT"):
        validate_claims_from_evidence(payload, implementation_commit=commit)


def test_tamper_arm_order_metadata_vs_trace_is_rejected() -> None:
    payload, commit = _payload()
    result = next(item for item in payload["arm_seed_results"] if item["arm"] == "task01_last")
    result["task_order"] = list(ARMS["canonical_order"])
    _resign(payload)
    with pytest.raises(ValueError, match="ARM_ORDER_METADATA"):
        validate_claims_from_evidence(payload, implementation_commit=commit)


@pytest.mark.parametrize("kind", ["duplicate_result", "missing_checkpoint"])
def test_duplicate_or_missing_coverage_is_rejected(kind: str) -> None:
    payload, commit = _payload()
    if kind == "duplicate_result":
        payload["arm_seed_results"][-1] = copy.deepcopy(payload["arm_seed_results"][0])
        match = "ARM_SEED_COVERAGE"
    else:
        payload["arm_seed_results"][0]["checkpoints"].pop()
        match = "CHECKPOINT_COVERAGE"
    _resign(payload)
    with pytest.raises(ValueError, match=match):
        validate_claims_from_evidence(payload, implementation_commit=commit)


def test_forged_canonical_prefix_record_is_rejected() -> None:
    payload, commit = _payload()
    result = next(
        item
        for item in payload["arm_seed_results"]
        if item["arm"] == "canonical_order" and item["seed"] == 17
    )
    result["prefix_equivalence"]["control"]["trained_canonical_sha256"] = "sha256:forged"
    _resign(payload)
    with pytest.raises(ValueError, match="CANONICAL_PREFIX_BINDING"):
        validate_claims_from_evidence(payload, implementation_commit=commit)


def test_coherent_raw_and_prefix_forge_is_rejected_by_frozen_anchor() -> None:
    payload, commit = _payload()
    result = next(
        item
        for item in payload["arm_seed_results"]
        if item["arm"] == "canonical_order" and item["seed"] == 17
    )
    update = result["updates"][0]
    update["operator_loss"] = float(update["operator_loss"]) + 0.25
    update["total_loss"] = validator._float32_total(
        float(update["operator_loss"]),
        update["arg1_pointer_loss"],
        update["arg2_pointer_loss"],
    )
    checkpoint3 = next(item for item in result["checkpoints"] if item["epoch"] == 3)
    checkpoint3["trained_canonical_sha256"] = "sha256:coherently-forged-trained"
    checkpoint3["optimizer_canonical_sha256"] = "sha256:coherently-forged-optimizer"
    forged = validator._actual_prefix(result)
    result["prefix_equivalence"]["control"] = copy.deepcopy(forged)
    result["prefix_equivalence"]["arm_prefix"] = copy.deepcopy(forged)
    _resign(payload)
    with pytest.raises(ValueError, match="FROZEN_CONTROL_ANCHOR"):
        validate_claims_from_evidence(payload, implementation_commit=commit)


def test_loss_decomposition_tamper_is_rejected() -> None:
    payload, commit = _payload()
    update = payload["arm_seed_results"][0]["updates"][1]
    update["total_loss"] = float(update["total_loss"]) + 0.5
    _resign(payload)
    with pytest.raises(ValueError, match="LOSS_DECOMPOSITION"):
        validate_claims_from_evidence(payload, implementation_commit=commit)


def test_coherent_pointer_loss_tamper_is_rejected_by_applicability() -> None:
    payload, commit = _payload()
    update = payload["arm_seed_results"][0]["updates"][0]
    assert update["arg1_target_count"] == 0
    update["arg1_pointer_loss"] = 0.25
    update["total_loss"] = validator._float32_total(
        float(update["operator_loss"]),
        0.25,
        update["arg2_pointer_loss"],
    )
    _resign(payload)
    with pytest.raises(ValueError, match="ARG1_APPLICABILITY"):
        validate_claims_from_evidence(payload, implementation_commit=commit)


def test_stale_cross_arm_rescue_delta_is_rejected() -> None:
    payload, commit = _payload()
    payload["cross_arm_rescue_deltas"]["task01_middle"]["by_seed"]["17"][
        "first_position0_operator_rescue_update_delta_vs_canonical"
    ] += 3
    _resign(payload)
    with pytest.raises(ValueError, match="CROSS_ARM_DELTA"):
        validate_claims_from_evidence(payload, implementation_commit=commit)


def test_changed_raw_checkpoint_claim_is_rejected() -> None:
    payload, commit = _payload()
    result = payload["arm_seed_results"][0]
    position = result["checkpoints"][2]["teacher_forced"][1]["positions"][0]
    position["predicted_operator"] = "END"
    position["operator_correct"] = False
    position["joint_step_correct"] = False
    _resign(payload)
    with pytest.raises(ValueError, match="CHECKPOINT_TEACHER_SUMMARY"):
        validate_claims_from_evidence(payload, implementation_commit=commit)


def test_real_frozen_control_reconstruction_has_exact_nine_updates(
    _synthetic_frozen_anchor,
) -> None:
    dataset = generate_train_only()
    rows = sorted(list(dataset["train"]), key=lambda row: row["task_id"])
    control = _synthetic_frozen_anchor(
        rows,
        seed=17,
        dataset_hash=dataset["frozen_dataset_lineage_hash"],
    )
    assert len(control["updates"]) == 9
    assert [update["task_id"] for update in control["updates"]] == list(TASKS) * 3
    assert all(set(update) == set(validator.PREFIX_TRACE_FIELDS) for update in control["updates"])
