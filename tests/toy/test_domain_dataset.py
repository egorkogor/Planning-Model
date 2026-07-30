from pathlib import Path

import pytest

from planner_toy.dataset import generate, write_dataset
from planner_toy.domain import Task, applicable, apply_action, shortest_plan, validate_state


def test_canonical_state_goal_and_preconditions():
    blocks = ("@B0", "@B1")
    state = validate_state(
        blocks, (("CLEAR", "@B1"), ("ON", "@B1", "@B0"), ("ON_TABLE", "@B0"), ("HAND_EMPTY",))
    )
    assert state == tuple(sorted(state))
    assert applicable(blocks, state, ("UNSTACK", "@B1", "@B0"))
    assert not applicable(blocks, state, ("PICK_UP", "@B0"))
    with pytest.raises(ValueError):
        apply_action(blocks, state, ("PICK_UP", "@B0"))


def test_deterministic_apply_and_shortest_bfs():
    row = generate()["validation"][0]
    task = Task(
        tuple(row["blocks"]), tuple(map(tuple, row["initial"])), tuple(map(tuple, row["goal"]))
    )
    first = shortest_plan(task)
    assert first == shortest_plan(task)
    state = task.initial
    for action in first:
        state = apply_action(task.blocks, state, action)
    assert set(task.goal) <= set(state)


def test_dataset_is_stable_disjoint_and_byte_reproducible(tmp_path: Path):
    left = generate()
    right = generate()
    assert left == right
    train = {row["canonical_task_hash"] for row in left["train"]}
    validation = {row["canonical_task_hash"] for row in left["validation"]}
    assert train.isdisjoint(validation)
    assert all(row["task_id"].startswith("bw-") for row in left["train"] + left["validation"])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_dataset(a)
    write_dataset(b)
    assert a.read_bytes() == b.read_bytes()
