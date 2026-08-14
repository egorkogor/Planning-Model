"""Versioned train-only materialization for the A2 learnability diagnostic."""

from __future__ import annotations

from typing import Final

from .canonical import sha256
from .domain import Task, canonical_facts, shortest_plan

FROZEN_DATASET_LINEAGE_HASH_V1: Final = (
    "sha256:60e4ce06d6cfc90dc467fb4e82b2eb71cf2d92d37471eee3aeda64f864c541df"
)
TRAIN_ONLY_SCHEMA_VERSION: Final = "toy-a2-train-only-dataset/1.0"
TRAIN_SPLIT_HASH_VERSION: Final = "toy-a2-evaluated-train-split-hash/1.0"
_TRAIN_ONLY_SERIAL_ORDER_V1: Final = (1, 3, 2)
_TASK_SPECS_V1: Final = {1: (1, (0,)), 2: (2, (0, 1)), 3: (2, (1, 0))}


def _tower_state(blocks: tuple[str, ...], order: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    facts = {("HAND_EMPTY",), ("ON_TABLE", order[-1]), ("CLEAR", order[0])}
    facts |= {("ON", order[i], order[i + 1]) for i in range(len(order) - 1)}
    return canonical_facts(facts)


def _build_train_row(serial: int) -> dict:
    """Build one train row; held-out serials have no specification here."""
    try:
        n_blocks, initial_indices = _TASK_SPECS_V1[serial]
    except KeyError as exc:
        raise ValueError(f"unknown train-only task serial: {serial}") from exc
    blocks = tuple(f"@B{i}" for i in range(n_blocks))
    initial_order = tuple(blocks[index] for index in initial_indices)
    goal_order = tuple(reversed(initial_order)) if n_blocks > 1 else initial_order
    initial = _tower_state(blocks, initial_order)
    goal = (("ON_TABLE", goal_order[-1]),) + tuple(
        ("ON", goal_order[index], goal_order[index + 1])
        for index in range(n_blocks - 1)
    )
    task = Task(blocks, initial, canonical_facts(goal), f"bw-{serial:08d}")
    plan = shortest_plan(task)
    return {**task.payload(), "task_id": task.task_id,
            "canonical_task_hash": task.canonical_hash,
            "oracle_work_plan": [list(action) for action in plan] + [["END"]]}


def generate_train_only(seed: int = 17) -> dict:
    """Materialize only frozen seed-17 train rows in historical lineage order."""
    if seed != 17:
        raise ValueError("TOY_TRAIN_ONLY_SEED_UNSUPPORTED")
    train = [_build_train_row(serial) for serial in _TRAIN_ONLY_SERIAL_ORDER_V1]
    train_task_ids = [row["task_id"] for row in train]
    evaluated_train_split_hash = sha256(
        {"schema_version": TRAIN_SPLIT_HASH_VERSION, "seed": seed, "train": train}
    )
    return {"schema_version": TRAIN_ONLY_SCHEMA_VERSION, "seed": seed, "train": train,
            "train_task_ids": train_task_ids,
            "frozen_dataset_lineage_hash": FROZEN_DATASET_LINEAGE_HASH_V1,
            "evaluated_train_split_hash": evaluated_train_split_hash}
