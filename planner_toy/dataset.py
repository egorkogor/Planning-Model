"""Reproducible seed-17 toy dataset, disjoint from sealed data."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .canonical import canonical_bytes, sha256
from .domain import Task, canonical_facts, shortest_plan


def _tower_state(blocks: tuple[str, ...], order: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    facts = {("HAND_EMPTY",), ("ON_TABLE", order[-1]), ("CLEAR", order[0])}
    facts |= {("ON", order[index], order[index + 1]) for index in range(len(order) - 1)}
    return canonical_facts(facts)


def generate(seed: int = 17) -> dict:
    rng = random.Random(seed)
    tasks = []
    serial = 1
    for n_blocks in range(1, 4):
        blocks = tuple(f"@B{i}" for i in range(n_blocks))
        orders = [tuple(blocks)]
        if n_blocks > 1:
            reversed_order = tuple(reversed(blocks))
            orders.append(reversed_order)
        for initial_order in orders:
            goal_order = tuple(reversed(initial_order)) if n_blocks > 1 else initial_order
            initial = _tower_state(blocks, initial_order)
            goal = (("ON_TABLE", goal_order[-1]),) + tuple(
                ("ON", goal_order[index], goal_order[index + 1]) for index in range(n_blocks - 1)
            )
            task = Task(blocks, initial, canonical_facts(goal), f"bw-{serial:08d}")
            plan = shortest_plan(task)
            tasks.append(
                {
                    **task.payload(),
                    "task_id": task.task_id,
                    "canonical_task_hash": task.canonical_hash,
                    "oracle_work_plan": [list(action) for action in plan] + [["END"]],
                }
            )
            serial += 1
    rng.shuffle(tasks)
    train, validation = tasks[:3], tasks[3:]
    return {
        "schema_version": "toy-a2-dataset/1.0",
        "seed": seed,
        "confirmatory": False,
        "train": train,
        "validation": validation,
        "dataset_hash": sha256({"seed": seed, "train": train, "validation": validation}),
    }


def write_dataset(path: Path) -> dict:
    dataset = generate()
    data = canonical_bytes(dataset) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError("existing toy dataset differs from deterministic generation")
    path.write_bytes(data)
    return json.loads(data)


def task_from_row(row: dict) -> Task:
    return Task(
        tuple(row["blocks"]),
        tuple(tuple(fact) for fact in row["initial"]),
        tuple(tuple(fact) for fact in row["goal"]),
        row["task_id"],
    )
