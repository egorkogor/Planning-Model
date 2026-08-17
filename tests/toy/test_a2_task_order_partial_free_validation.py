from __future__ import annotations

import copy

import pytest

import planner_toy.a2_sufficient_budget_task_order_validator as validator
from planner_toy.dataset import task_from_row
from planner_toy.domain import goal_satisfied, validate_state
from planner_toy.train_only_dataset import generate_train_only

ARM = "canonical_order"
SEED = 17
EPOCH = 17
TASK_ID = "bw-00000003"


def _task03_row() -> dict:
    rows = list(generate_train_only()["train"])
    return next(row for row in rows if row["task_id"] == TASK_ID)


def _initial_goal_satisfied(row: dict) -> bool:
    task = task_from_row(row)
    state = validate_state(task.blocks, task.initial)
    return goal_satisfied(state, task.goal)


def _realistic_nonterminal_partial(row: dict) -> list[list[str]]:
    plan = copy.deepcopy(row["oracle_work_plan"][:-1])
    assert plan
    assert plan[-1] != ["END"]
    assert any(step[0] == "UNSTACK" for step in plan)
    return plan


def _record(row: dict, plan: list[list[str]], *, length: int, success: bool) -> dict:
    return {
        "task_id": row["task_id"],
        "initial_goal_satisfied": _initial_goal_satisfied(row),
        "predicted_plan": plan,
        "predicted_plan_length": length,
        "exact_plan_match": plan == row["oracle_work_plan"],
        "final_goal_success": success,
    }


def test_nonterminal_partial_plan_keeps_nonzero_predicted_length() -> None:
    row = _task03_row()
    plan = _realistic_nonterminal_partial(row)

    initial, success, length = validator._free_goal_success(row, plan)

    assert initial is False
    assert success is False
    assert length == len(plan) > 0
    validator._validate_free_record(
        _record(row, plan, length=len(plan), success=False),
        row,
        arm=ARM,
        seed=SEED,
        epoch=EPOCH,
    )


def test_nonterminal_partial_plan_rejects_forged_zero_length() -> None:
    row = _task03_row()
    plan = _realistic_nonterminal_partial(row)

    with pytest.raises(ValueError, match="A2_ORDER_VALIDATOR_FREE_LENGTH"):
        validator._validate_free_record(
            _record(row, plan, length=0, success=False),
            row,
            arm=ARM,
            seed=SEED,
            epoch=EPOCH,
        )


def test_nonterminal_partial_plan_cannot_be_promoted_to_goal_success() -> None:
    row = _task03_row()
    plan = _realistic_nonterminal_partial(row)

    with pytest.raises(ValueError, match="A2_ORDER_VALIDATOR_FREE_GOAL"):
        validator._validate_free_record(
            _record(row, plan, length=len(plan), success=True),
            row,
            arm=ARM,
            seed=SEED,
            epoch=EPOCH,
        )


def test_nonterminal_parse_failure_still_counts_all_predicted_actions() -> None:
    row = _task03_row()
    plan = _realistic_nonterminal_partial(row)
    malformed = [copy.deepcopy(plan[0]), ["UNSTACK", "not-a-block", "also-not-a-block"]]

    initial, success, length = validator._free_goal_success(row, malformed)

    assert initial is False
    assert success is False
    assert length == 2
    validator._validate_free_record(
        _record(row, malformed, length=2, success=False),
        row,
        arm=ARM,
        seed=SEED,
        epoch=EPOCH,
    )
